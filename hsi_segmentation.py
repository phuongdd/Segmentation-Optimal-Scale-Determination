"""."""

import numpy as np
import pandas as pd
import itertools
from pathlib import Path
from functools import partial

import scipy.signal
import scipy.ndimage
import scipy.stats

import h5py
import deepdish as dd

from tqdm.auto import tqdm

# import faiss

import skimage.util
import sklearn.cluster
import sklearn.preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV


import skimage.color
import skimage.exposure
import skimage.segmentation
import skimage.filters
import skimage.morphology

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import property_manager

from IPython.core.display import display, HTML

from hsi_dataset import HSIDataset


class HSISegmentation:
    """Create segmentation and perform calculations."""

    def __init__(self, dataset: HSIDataset):
        """Perform segmentation on dataset (HSIDataset) using several methods
        (kmeans, meanshift, knn, watershed, watershed_combined_grads)
        Arguments:
            dataset {HSIDataset} -- Input dataset used for processing.
                                    Must be compatible with HSIDataset
        """
        self.dataset = dataset
        self.results_dir = self.dataset.base_dir / 'results'
        self.models_dir = self.dataset.base_dir / 'models'
        self.results_file = f"{self.results_dir /   self.dataset.base_name}.h5"
        print(f'Opening:{self.results_file}')
        self.results = h5py.File(name=self.results_file, mode='a')
        self.stats_cache = {}

    # --------------------------------------------------------------------------------------------------------------- #
    @property
    def list_segments(self):
        keys = []

        def find_segments(key, item):
            if 'segments' in key:
                keys.append(item.name)
            else:
                return None
        try:
            self.results.visititems(find_segments)
        except Exception as e:
            print(e)

        return keys

    @property
    def list_stats(self):
        keys = []

        def find_segments(key, item):
            if 'stats' in key:
                keys.append(item.name)
            else:
                return None
        self.results.visititems(find_segments)
        return keys

    @property
    def list_all(self):
        keys = []

        def find_segments(key, item):
            keys.append(item.name)
            return None
        self.results.visititems(find_segments)
        return keys

    # --------------------------------------------------------------------------------------------------------------- #
    @property
    def algorithms(self):
        return [self.kmeans.__name__,
                self.meanshift.__name__,
                self.knn.__name__,
                self.watershed_traditional.__name__,
                self.watershed.__name__
                ]

    # def kmeans(self, feature_vector, k):
    #     if k == 0:
    #         return np.zeros(shape=(self.dataset.n_rows, self.dataset.n_cols), dtype=int)

    #     arr = feature_vector
    #     X = arr.reshape(self.dataset.n_pixels, -1).astype(np.float32)
    #     n_dim = X.shape[1]
    #     kmeans = faiss.Kmeans(d=n_dim, k=k, niter=50,  verbose=False)
    #     kmeans.train(X)
    #     _, kmeans_segments = kmeans.index.search(X, k=1)
    #     kmeans_segments = kmeans_segments.reshape(self.dataset.n_rows, self.dataset.n_cols, -1).squeeze()
    #     return kmeans_segments

    def meanshift(self, feature_vector, k, bandwidth=None):
        if k == 0:
            return np.zeros(shape=(self.dataset.n_rows, self.dataset.n_cols), dtype=int)

        arr = feature_vector
        X = arr.reshape(self.dataset.n_pixels, -1).astype(np.float32)
        np.random.seed(0)
        seeds_idx = np.random.choice(len(X), k)
        if bandwidth is None:
            bandwidth = X[seeds_idx].std()
            # bandwidth = sklearn.cluster.estimate_bandwidth(X=X[seeds_idx], n_jobs=-1)

        meanshift_clusterer = sklearn.cluster.MeanShift(n_jobs=-1, bin_seeding=False, bandwidth=bandwidth)
        meanshift_clusterer.fit(X[seeds_idx])
        meanshift_segments = meanshift_clusterer.predict(X=X)
        meanshift_segments = meanshift_segments.reshape(self.dataset.n_rows, self.dataset.n_cols)
        return meanshift_segments

    # def knn(self, feature_vector, k):
    #     if k == 0:
    #         return np.zeros(shape=(self.dataset.n_rows, self.dataset.n_cols), dtype=int)

    #     arr = feature_vector
    #     X = arr.reshape(self.dataset.n_pixels, -1).astype(np.float32)
    #     n_dim = X.shape[1]
    #     seeds = np.random.choice(self.dataset.n_pixels, k)
    #     index = faiss.IndexFlatL2(n_dim)   # build the index
    #     index.add(X)
    #     index.train(X)

    #     _, knn = index.search(X[seeds, :], k=int(self.dataset.n_pixels/k))

    #     knn_segments = np.zeros(shape=(self.dataset.n_pixels))
    #     for label, segment in enumerate(knn):
    #         knn_segments[segment] = label
    #     knn_segments = knn_segments.reshape(self.dataset.n_rows, self.dataset.n_cols)
    #     return knn_segments

    def watershed_traditional(self, feature_vector, markers, compactness=1E-5):
        """Compute watershed segmentation over a combined gradient image.


        """
        if markers == 0:
            return np.zeros(shape=(self.dataset.n_rows, self.dataset.n_cols), dtype=int)

        arr = feature_vector
        n_bands = self.dataset.n_bands
        joined_segments = np.zeros(shape=(self.dataset.n_rows, self.dataset.n_cols))
        for b in tqdm(range(n_bands), desc="watershed layer:"):
            band = arr[:, :, b]
            grads = skimage.filters.sobel(band)
            segmentation = skimage.segmentation.watershed(grads, markers=markers, compactness=compactness)
            joined_segments = skimage.segmentation.join_segmentations(joined_segments, segmentation)
        segmentation = joined_segments
        return segmentation

    def watershed(self, feature_vector, markers, compactness=1E-5, aggregator_function="sum"):
        """Compute watershed segmentation over a combined gradient image.

        """
        if markers == 0:
            return np.zeros(shape=(self.dataset.n_rows, self.dataset.n_cols), dtype=int)

        arr = feature_vector
        n_bands = arr.shape[2]
        stacked_grads = []
        for b in tqdm(range(n_bands), desc="watershed layer:"):
            grads = skimage.filters.sobel(arr[:, :, b])
            stacked_grads.append(grads)
        stacked_grads = np.stack(stacked_grads, axis=-1)

        if aggregator_function == "mean":
            sobel = np.mean(stacked_grads, axis=2)
        else:
            sobel = np.sum(stacked_grads, axis=2)

        # sobel_v = scipy.ndimage.sobel(arr, axis=0)
        # sobel_h = scipy.ndimage.sobel(arr, axis=1)

        # sobel_v = func(sobel_v, axis=2, keepdims=False, dtype=np.uint32)
        # sobel_h = func(sobel_h, axis=2, keepdims=False, dtype=np.uint32)
        # sobel = sobel_h + sobel_v

        segmentation = skimage.segmentation.watershed(sobel, markers=markers, compactness=compactness)
        return segmentation

    # --------------------------------------------------------------------------------------------------------------- #
    # Classifiers
    @property_manager.cached_property
    def rf_classifier(self):
        filename = f'{self.models_dir / self.base_name}.rf_classifier.h5'
        if Path(filename).exists():
            print(f"Loading classifier file {filename}.")
            classifier = dd.io.load(filename)
        else:
            classifier = self.random_forest(mode="train")
        return classifier

    def random_forest(self, mode="train"):
        """
        Classify and Train using Random Forests.

        mode: 'train'  or 'predict'
        """
        if mode == "train":
            X = self.dataset.hsi.reshape(self.dataset.n_rows*self.dataset.n_cols, -1)
            y = self.dataset.trainingset[1].reshape((self.dataset.n_rows*self.dataset.n_cols)).squeeze()
            train_pixels = (y > 0)

            X = X[train_pixels, :]
            y = y[train_pixels]

            n_estimators = [x for x in np.linspace(start=1000, stop=1000, num=1, dtype=int)]
            max_depth = [x for x in np.linspace(1000, 1000, num=1, dtype=int)] + [None]
            random_grid = {'n_estimators': n_estimators,
                           # 'max_features': ['auto', 'sqrt'],
                           'max_depth': max_depth,
                           # 'min_samples_split': [2, 5, 10],
                           # 'min_samples_leaf': [1, 2, 4],
                           # 'bootstrap': [True, False]
                           }

            rfr_grid = RandomizedSearchCV(estimator=RandomForestClassifier(),
                                          param_distributions=random_grid,
                                          n_iter=10, cv=2, verbose=2,
                                          n_jobs=-1)

            print("Fit the random search model for the grid search...")
            rfr_grid.fit(X=X, y=y)

            classifier = rfr_grid.best_estimator_
            print(f"Fit the best estimator: {classifier}")
            classifier.fit(X=X, y=y)

            filename = f'{self.models_dir / self.base_name}.rf_classifier.h5'
            print(f"Saving classifier file {filename}.")
            dd.io.save(filename, classifier)

            return classifier
        elif mode == "test":
            X = self.dataset.hsi.reshape(self.dataset.n_rows*self.dataset.n_cols, -1)
            labels, y = self.dataset.testset

            y = y.reshape((self.dataset.n_rows*self.dataset.n_cols, -1))

            test_pixels = (y > 0).squeeze()
            X = X[test_pixels, :]
            y = y[test_pixels, :]

            predictions = self.rf_classifier.predict(X=X)

            y = y.reshape(-1)
            predictions = predictions.reshape(-1)

            confusion_matrix = sklearn.metrics.confusion_matrix(y, predictions)
            confusion_matrix = pd.DataFrame(confusion_matrix, columns=labels, index=labels)

            results = np.zeros(shape=(self.dataset.n_pixels))
            results[test_pixels] = predictions
            results = results.reshape(self.dataset.n_rows, self.dataset.n_cols)
            return results, confusion_matrix
        else:  # mode == "predict" or anything else.
            X = self.dataset.hsi.reshape(self.dataset.n_rows*self.dataset.n_cols, -1)
            predicted = self.rf_classifier.predict(X)
            predicted = predicted.reshape(self.dataset.n_rows, self.dataset.n_cols)
            return predicted

    def svm(self, mode="train"):
        """Classify and Train using SVM.

        mode: 'train'  or 'predict'
        """
        return None

    def gbm(self, mode="train"):
        """
        Classify and Train using gradient boostes trees.

        mode: 'train'  or 'predict'
        """
        return None
        pass

    # --------------------------------------------------------------------------------------------------------------- #
    def run(self, algorithm_name, hparams, feature_vector_name='HSI', force=True):
        """"

        Arguments:
            feature_vector_name : 'HSI', 'PCA', 'RGB', 'HSY_XY', 'RGB_XY', 'PCA_XY' (must be available in self.feature_vetors)
            algorithm_name: 'kmeans', 'knn', 'meanshift', 'watershed_combined_grads', 'watershed'
            h_params: dict -- { algorithm_name : list of params}
                e.g.
                { 'knn': [10, 50],
                  'kmeans': [10, 50]
                 }

        Returns:
            True if succesfull.
            Save segmentation into HDF5 file.
        """

        algorithm = partial(eval(f'self.{algorithm_name}'))
        algorithm_name = algorithm.func.__name__

        hparams_list = list(itertools.product(*hparams.values()))
        hparams_list = pd.DataFrame(hparams_list, columns=hparams.keys()).to_dict(orient='record')

        pbar = tqdm(hparams_list, desc=f'Segmentation {algorithm_name}')
        for params in pbar:

            _params = "/".join([str(v) for v in params.values()])
            base_key = f"{self.dataset.name}/{algorithm_name}/{_params}"
            key = f"{self.dataset.name}/{algorithm_name}/{_params}/segments"

            pbar.set_postfix(dataset=key, refresh=True)

            if base_key in self.results and force:
                del self.results[base_key]
            elif key in self.results:
                pbar.set_postfix(dataset=f"{key} skipped", refresh=True)
                continue

            feature_vector = self.dataset.feature_vectors[feature_vector_name]
            feature_vector[self.dataset.mask[:, :, -1]] = 1e5

            # Magic ]starts here then goes into inside the function (algorithm)
            segmentation = algorithm(feature_vector=feature_vector, **params)

            self.results.create_dataset(key, data=segmentation)
            if (int(100*(pbar.n/pbar.total)) % 10) == 0:
                self.results.flush()

        self.results.flush()
        return True

    def stats(self, algorithm_name, stat_name, force=False, fix_small=True, scales=None):
        prefix = f"{self.dataset.name}/{algorithm_name}"

        if self.stats_cache.get(prefix, {}).get(stat_name, None) is not None and not force:
            df = self.stats_cache[prefix][stat_name]
            n_df = self.stats_cache[prefix][f'n_{stat_name}']
            return df.copy(), n_df.copy()

        stats_functions = {
            "max": scipy.ndimage.maximum,
            "min": scipy.ndimage.minimum,
            "mean": scipy.ndimage.mean,
            "median": scipy.ndimage.median,
            "std": scipy.ndimage.standard_deviation,
            "variance": scipy.ndimage.variance,

            "mode": partial(
                scipy.ndimage.labeled_comprehension,
                func=lambda arr: scipy.stats.mode,
                out_dtype=np.float32,
                default=np.nan
            ),

            "size": partial(
                scipy.ndimage.labeled_comprehension,
                func=len,
                out_dtype=np.int32,
                default=np.nan
            ),
            "coef_variation": partial(
                scipy.ndimage.labeled_comprehension,
                func=lambda arr: np.std(arr)/np.mean(arr),
                out_dtype=np.float32,
                default=np.nan
            )
        }

        # ----------------------------------------------------------------------------------------------------------- #
        # Remove Outliers Function
        def remove_outliers(stat):
            series = pd.Series(stat)
            series.dropna(inplace=True)

            q25 = series.quantile(0.25)
            q75 = series.quantile(0.75)
            iqr = q75 - q25

            cut_off = iqr
            low_bound = q25 - cut_off
            up_bound = q75 + cut_off

            # print(f"{q25:.4f}, {iqr:.4f}, {q75:.4f}, {low_bound:.4f}, {up_bound:.4f}")

            idxs_outliers = (series < low_bound) | (series > up_bound)
            idxs_non_outliers = (series >= low_bound) & (series <= up_bound)

            labels_outliers = series[idxs_outliers].index
            labels_non_outliers = series[idxs_non_outliers].index

            return q25, q75, iqr, labels_non_outliers, labels_outliers
        # ----------------------------------------------------------------------------------------------------------- #

        df = pd.DataFrame()
        n_df = pd.DataFrame()

        pbar = tqdm(self.results[prefix].items(), desc=f'Stats {stat_name} {algorithm_name}')
        for key, item in pbar:
            param = int(key)
            if scales is not None and param not in scales:
                continue

            if 'watershed' in algorithm_name:
                segments_key = f"{item.name}/1e-05/segments"
            else:
                segments_key = f"{item.name}/segments"

            stat_key = f"{item.name}/stats/{stat_name}"
            n_stat_key = f"{item.name}/stats/n_{stat_name}"

            if stat_key not in self.results or force:
                pbar.set_postfix(dataset=item.name, refresh=True)
                segmentation = self.results[segments_key][()]

                if stat_key in self.results:
                    del self.results[stat_key]
                if n_stat_key in self.results:
                    del self.results[n_stat_key]

                pbar.set_postfix(dataset=f"{item.name} | stat function: {stat_name}", refresh=True)
                hsi = np.mean(a=self.dataset.hsi, axis=2)
                func = stats_functions['coef_variation']

                pbar.set_postfix(dataset=f"re-label of segments {item.name}", refresh=True)

                if fix_small and 'watershed' not in algorithm_name:
                    pbar.set_postfix(dataset=f"{item.name} (fixing small objects)", refresh=True)
                    segmentation = skimage.morphology.closing(segmentation, skimage.morphology.square(3))
                    segmentation = skimage.measure.label(segmentation, connectivity=1)

                segmentation += 100
                segmentation[self.dataset.mask[:, :, -1]] = -1  # filter out masked in the next line
                segments_labels = np.unique(segmentation[segmentation >= 0])
                n_segments = len(segments_labels)

                pbar.set_postfix(dataset=f"stats - {item.name}", refresh=True)
                stat = func(input=hsi, labels=segmentation, index=segments_labels)

                pbar.set_postfix(dataset=f"stats for ouliers removal - {item.name}", refresh=True)
                q25, q75, iqr, labels_non_outliers, labels_outliers = remove_outliers(stat)

                n_stat = stat[labels_non_outliers]

                stat = stat[stat > 0]
                n_stat = n_stat[n_stat > 0]

                # TODO: Fix using stats only over non zero indices
                # mask_outliers = np.isin(segmentation, labels_outliers)
                # segments_non_outliers = np.where(mask_outliers, 0, segmentation)
                # n_stat = func(input=hsi, labels=segments_non_outliers, index=np.unique(labels_non_outliers))

                pbar.set_postfix(dataset=f"{item.name} saving data into dataset hdf5", refresh=True)

                self.results.create_dataset(stat_key, data=stat)
                self.results.create_dataset(n_stat_key, data=n_stat)

                pbar.set_postfix(dataset=f"{item.name} saving txt file with stats", refresh=True)
                string_out = f"q25 = {q25}\n"
                string_out += f"q75 = {q75}\n"
                string_out += f"iqr = {iqr}\n"
                string_out += f"n_segments = {n_segments}\n"
                string_out += f"n_segments_outlier = {len(np.unique(labels_outliers))}\n"
                string_out += f"n_segments_non_outliers = {len(np.unique(labels_non_outliers))}\n"
                string_out += f"mean_cv = {np.mean(stat)}\n"
                string_out += f"non_outlier_mean_cv = {np.mean(n_stat)}\n"

                stat_file_name = (Path(__file__).parent / f"results/{stat_key.replace('/', '_')}").with_suffix(f'.txt')
                open(stat_file_name, 'w').write(string_out)
            else:
                pbar.set_postfix(dataset=f'{stat_key} (cached)', refresh=True)
                stat = self.results[stat_key][()]
                n_stat = self.results[n_stat_key][()]

            pbar.set_postfix(dataset=f"{item.name} (building dataframe)", refresh=True)
            df = df.merge(pd.DataFrame(stat, columns=[param]), how='outer', left_index=True, right_index=True)
            n_df = n_df.merge(pd.DataFrame(n_stat, columns=[param]),
                              how='outer', left_index=True, right_index=True)
            df.sort_index(axis=1, inplace=True)
            n_df.sort_index(axis=1, inplace=True)

        csv_filename = Path(__file__).parent / f'results/CV_{self.dataset.name}_{algorithm_name}.csv'
        _tmpdf = pd.DataFrame(df.mean(axis=0), columns=['CV'])
        _tmpdf['RoC'] = _tmpdf['CV'].pct_change()

        _tmpdf['nCV'] = n_df.mean()
        _tmpdf['nRoC'] = _tmpdf['nCV'].pct_change()
        _tmpdf.to_csv(csv_filename)

        df = df.melt(var_name='parameter').dropna()
        df['group'] = 'CV'
        n_df = n_df.melt(var_name='parameter').dropna()
        n_df['group'] = 'nCV'

        df_final = pd.concat([df, n_df])
        self.stats_cache[prefix] = df_final

        return df_final

    def post_segmentation_stats(self, alg_name, param, debug=False):
        if 'watershed' in alg_name:
            h5_key = f'{self.dataset.name}/{alg_name}/{param}/1e-05/segments'
        else:
            h5_key = f'{self.dataset.name}/{alg_name}/{param}/segments'

        classes, labelled_shapes = self.dataset.trainingset
        classes_names = [None]*len(classes)
        for class_name, class_id in classes.items():
            classes_names[class_id] = class_name

        n_r_polygons = len(np.unique(labelled_shapes))
        r_polygons = skimage.measure.label(labelled_shapes, connectivity=2)
        index = np.unique(r_polygons[r_polygons > 0])

        segmentation = self.results[h5_key][()]
        segmentation[self.dataset.mask[:, :, -1]] = 0

        segmentation = skimage.morphology.closing(image=segmentation, selem=skimage.morphology.square(3))
        segmentation = skimage.measure.label(segmentation, connectivity=2)

        metrics_columns = ['algorithm', 'segment_id', 'polygon_id', 'polygon_label',
                           'union', 'intersection', 'area_s', 'area_r',
                           'area_intersection', 'area_union', 'OS', 'US', 'ED', 'IoU'
                           ]
        metrics = []

        pbar = tqdm(range(n_r_polygons), desc=alg_name)

        def segmentation_metrics(r, pos_r):
            r_id = np.unique(r)[-1]
            r_mask = np.zeros(shape=(self.dataset.n_rows*self.dataset.n_cols), dtype=int)
            r_mask[pos_r] = 1

            r_area = len(pos_r)  # All the same: len(r), len(pos_r), len(r_mask[r_mask == 1])
            r_label = labelled_shapes.ravel()[pos_r][-1] # Label identification can be the value of any pixel from the polygon
            r_label_name = classes_names[r_label]

            pbar.set_postfix(polygon=f"{r[-1]}, label={r_label}({r_label_name})", refresh=True)
            pbar.update(1)

            def inner(s, pos_s):
                s_id = np.unique(s)[-1]
                s_area = len(s)  # len(s), len(pos_s), len(s_mask[s_mask == 1])
                if s_area < 100:
                    return 0

                s_mask = np.zeros(shape=(self.dataset.n_rows*self.dataset.n_cols), dtype=int)
                s_mask[pos_s] = 1

                union = np.logical_or(r_mask, s_mask)
                intersection = np.logical_and(r_mask, s_mask)

                union_area = len(union[union > 0])
                intersection_area = len(intersection[intersection > 0])
                iou = (intersection_area / union_area)

                os_i_j = 1 - (intersection_area / r_area)
                us_i_j = 1 - (intersection_area / s_area)
                ed_i_j = ((os_i_j**2 + us_i_j**2)**(1/2)) / 2


                # --- Debug -------- #
                if debug:
                    print(f"Polygon label: {r_label_name}, segment_id: {s_id}")
                    print(f"intersection_area: {intersection_area}, union_area: {union_area}, r_area: {r_area}, s_area: {s_area}")
                    print(f"os_i_j: {os_i_j}, us_i_j: {us_i_j}, ed_i_j: {ed_i_j}")

                    fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))
                    axs[0].imshow(r_mask.reshape(self.dataset.n_rows, self.dataset.n_cols), cmap='binary')
                    axs[1].imshow(s_mask.reshape(self.dataset.n_rows, self.dataset.n_cols), cmap='binary')
                    axs[2].imshow(intersection.reshape(self.dataset.n_rows, self.dataset.n_cols), cmap='binary')
                    axs[3].imshow(union.reshape(self.dataset.n_rows, self.dataset.n_cols), cmap='binary')
                    axs[0].set_title('Polygon')
                    axs[1].set_title('Segment')
                    axs[2].set_title('intersection')
                    axs[3].set_title('union')
                    for ax in axs.ravel():
                        ax.grid('off')
                        ax.axis('off')
                    plt.show()
                    plt.close()
                    display(HTML('<hr>'))

                # --- Debug -------- #

                metric_record = [alg_name, s_id, r_id, r_label,
                                 union, intersection,
                                 s_area, r_area, intersection_area, union_area,
                                 os_i_j, us_i_j, ed_i_j, iou
                                 ]
                metric_record = dict(zip(metrics_columns, metric_record))

                metrics.append(metric_record)
                pbar.set_postfix(polygon=f"{r[-1]}, label={r_label}({r_label_name}) | segment_id:{s[-1]}", refresh=True)
                return 0

            # Collect the slice from the segmentation, that are inside the reference polygon
            segment_unique_labels = np.unique(segmentation.ravel()[pos_r])
            segment_unique_labels = segment_unique_labels[segment_unique_labels > 0]
            if len(segment_unique_labels) == 0:
                return 0

            # Walk through segment ids to collect basic data
            _ = scipy.ndimage.labeled_comprehension(input=segmentation,
                                                    labels=segmentation,
                                                    index=segment_unique_labels,
                                                    func=inner,
                                                    out_dtype=np.int,
                                                    default=0,
                                                    pass_positions=True
                                                    )
            return 0

        # ----------------------
        pbar.update(1)

        # fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
        # axs[0].imshow(labelled_shapes, cmap='jet')
        # axs[1].imshow(r_polygons, cmap='jet')
        # for ax in axs.ravel():
        #     ax.grid('off')
        #     ax.axis('off')
        # plt.show()
        # plt.close()

        # display(labelled_shapes.shape, r_polygons.shape)
        # display(labelled_shapes, r_polygons)
        # print(index)
        # for i in index:
        #     print(i)

        #     fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
        #     axs[0].imshow(labelled_shapes, cmap='jet')
        #     axs[1].imshow(r_polygons, cmap='jet')
        #     axs[2].imshow(np.where(r_polygons==i,1,0), cmap='binary')
        #     for ax in axs.ravel():
        #         ax.grid('off')
        #         ax.axis('off')
        #     plt.show()
        #     plt.close()

        _ = scipy.ndimage.labeled_comprehension(input=labelled_shapes,
                                                labels=r_polygons,
                                                index=index,
                                                func=segmentation_metrics,
                                                out_dtype=np.int,
                                                default=0,
                                                pass_positions=True
                                                )

        df = pd.DataFrame.from_records(metrics)
        df = df[metrics_columns]
        df['polygon_label_name'] = df.polygon_label.apply(lambda a: classes_names[a])
        return df

    # --------------------------------------------------------------------------------------------------------------- #

    def plot_charts(self,
                    algorithm_name,
                    show=False,
                    mean_roc=True,
                    big_boxplot=True,
                    boxplots_histograms=True,
                    segmentation=True,
                    stat_name='coef_variation',
                    force_stats=False,
                    fix_small=True,
                    scales=[]
                    ):

        df = self.stats(algorithm_name=algorithm_name, stat_name=stat_name, force=force_stats, fix_small=fix_small)
        prefix_title = f"{self.dataset.name} {algorithm_name}"
        df = df[df.value > 0]

        scales = np.sort(df.parameter.unique())
        #scales = scales[scales>=4]

        # Scales HERE !!!!
        if algorithm_name == 'kmeans' or algorithm_name == 'meanshift':
            idx = np.where((scales >= 0) & (scales <= 500))  # full
            # idx = np.where((scales >= 0) & (scales <= 25))   # detailed
            loc = 7
            xlabel = 'seeds (k)'
            scales = scales[idx]

        elif 'watershed' in algorithm_name:
            idx = np.where((scales >= 0) & (scales <= 10000))      # full
            # idx = np.where((scales >= 1000) & (scales <= 3000))  # detailed
            loc = 7
            xlabel = 'markers'
            scales = scales[idx]

        plot_title = f"{algorithm_name} k = [{min(scales)}-{max(scales)}]"
        # ------------------------------------------------------------------------------------------------------------ #

        def plot_segments(scales=[]):
            prefix = f"{self.dataset.name}/{algorithm_name}"

            # ------------------------------------------ #
            # Define SCALES here !!
            if len(scales) == 0:
                if 'watershed' in algorithm_name:
                    factor = 500
                    scales = [i for i in self.results[prefix].keys() if int(i) % factor == 0]
                else:
                    scales = [3,4,5,6,7,8,9,10,15,20,50,100]
            # ------------------------------------------ #

            nimages = len(scales)
            nrows = 4
            ncols = int((nimages*2)/4)

            fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15*ncols, 12*nrows))
            axs = axs.ravel()
            i = 0

            colors = np.linspace(0.1, 1, 100)
            np.random.shuffle(colors)
            cmap = plt.cm.colors.ListedColormap(plt.cm.tab20(colors))
            cmap.colors[0] = [0, 0, 0, 1]

            for scale in tqdm(scales, desc='Plot segments'):
                if i >= ncols*nrows:
                    break

                if 'watershed' in algorithm_name:
                    key = f'{self.dataset.name}/{algorithm_name}/{scale}/1e-05/segments'
                else:
                    key = f'{self.dataset.name}/{algorithm_name}/{scale}/segments'

                if key not in self.results:
                    continue

                segmentation = self.results[key][()]

                segmentation = skimage.morphology.closing(image=segmentation, selem=skimage.morphology.square(3))
                segmentation = skimage.measure.label(segmentation, connectivity=1)

                segmentation[self.dataset.mask[:, :, -1]] = 0
                boundaries = skimage.segmentation.mark_boundaries(
                    self.dataset.rgb_img_4display, segmentation, mode='outter', color=(1, 0, 0)
                )

                axs[i].imshow(boundaries)
                axs[i].set_title(f'boundaries param={scale}', fontsize=25)
                axs[i].axis("off")
                axs[i].grid('off')
                i += 1
                axs[i].imshow(segmentation, cmap=cmap)
                axs[i].set_title(f'segmentation param={scale}', fontsize=25)
                axs[i].axis("off")
                axs[i].grid('off')
                i += 1

            # plt.subplots_adjust(wspace=0.001, hspace=0.001)
            # fig.suptitle(algorithm_name, fontsize=13)
            plt.tight_layout()
            plt.savefig(f"{Path(__file__).parent}/results/segmentation_{self.dataset.name} {algorithm_name}.png")
            if show:
                plt.show()

            plt.close()

            return

        def plot_cv_mean_roc():
            _scales = scales[scales % 1 == 0]
            _scales = scales[scales <= 5_000]

            _scales = _scales.tolist()
            _df = df.query(f'parameter in {_scales}')

            cv_mean = _df.groupby(by=['parameter', 'group']).mean()
            cv_mean = pd.DataFrame(np.c_[cv_mean, cv_mean.pct_change()], columns=['CV', 'RoC'], index=cv_mean.index)
            cv_mean.sort_index(axis=0, inplace=True)
            cv_mean = cv_mean.reset_index('group')

            fig, ax = plt.subplots(figsize=(10, 8))
            ax2 = ax.twinx()

            l2 = ax.plot(cv_mean[cv_mean.group == 'nCV']['CV'], '-r', label='nCV', ms=0, lw=3, alpha=0.8)
            l1 = ax.plot(cv_mean[cv_mean.group == 'CV']['CV'], '-c', label='CV', ms=0, lw=3)

            l4 = ax2.plot(cv_mean[cv_mean.group == 'nCV']['RoC'], '--g', label='nRoC', ms=0, lw=2, alpha=0.8)
            l3 = ax2.plot(cv_mean[cv_mean.group == 'CV']['RoC'], '--b', label='RoC', ms=0, lw=2)

            #ax.axhline(y=0, linewidth=1, color='black', ls='-.', lw=2)
            #ax2.axhline(y=0, linewidth=1, color='black', ls='-.', lw=2)

            y_ticks = np.linspace(cv_mean['CV'].min(), cv_mean['CV'].max(), 8)
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(labels=y_ticks, rotation=90, va='center')
            ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:.2f}'))

            y2_ticks = np.linspace(cv_mean['RoC'].min(), cv_mean['RoC'].max(), 8)
            ax2.set_yticks(y2_ticks)
            ax2.set_yticklabels(labels=y2_ticks, rotation=90, va='center')
            ax2.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:.2f}'))

            lns = l1+l2+l3+l4
            labs = [l.get_label() for l in lns]
            ax2.legend(lns, labs, loc=loc)

            plt.title(plot_title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel('Coeficient of Variation', rotation=90)
            ax2.set_ylabel('Rate of Change', rotation=90)
            plt.tight_layout()
            plt.savefig(f"{Path(__file__).parent}/results/cv_roc.{prefix_title}.png")
            if show:
                plt.show()
            plt.close()

        def plot_combined_boxplot():
            if 'watershed' in algorithm_name:
                _scales = scales[scales % 200 == 0]
                labels_tick_factor = 500
            else:
                _scales = scales[scales % 2 == 0]
                labels_tick_factor = 10

            _scales = _scales.tolist()
            _df = df.query(f'parameter in {_scales}')

            fig, ax = plt.subplots(figsize=(12, 8))
            sns.boxplot(data=_df, x='parameter', y='value', hue='group', width=0.5)
            ax.set_xlabel('')
            texts = np.array([t.get_text() for t in ax.get_xticklabels()])
            for i, t in enumerate(texts):
                if int(t) % labels_tick_factor == 0:
                    texts[i] = t
                else:
                    texts[i] = ''
            ax.set_xticklabels(labels=texts)
            ax.set_ylabel('Coeficient of Variation')
            ax.set_xlabel(xlabel)
            ax.legend(title='', loc='best')
            plt.title(plot_title)
            plt.tight_layout()
            plt.savefig(f"{Path(__file__).parent}/results/combined_boxplot.{prefix_title}.png")
            if show:
                plt.show()
            plt.close()

        def plot_boxplots_histograms():
            if algorithm_name == 'kmeans':
                selection = [10]
                plot_title = f"{prefix_title} k = {selection}"
            elif algorithm_name == 'meanshift':
                selection = [10]
                plot_title = f"{algorithm_name} k = {selection}"
            elif 'watershed' in algorithm_name:
                selection = [3000]
                plot_title = f"{algorithm_name} markers = {selection}"

            for param in df.parameter.unique():
                if param not in selection:
                    continue

                df_param = df[df.parameter == param]
                # Individual BoxPlot
                for item in ['CV', 'nCV']:
                    _df = df_param[df_param.group == item]

                    # ----------------------------------------------------------------------------------------------- #
                    fig, (ax, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
                    sns.boxplot(data=_df, y='value', orient='h', ax=ax2)
                    ax.set_xlabel('Coeficient of Variation')
                    sns.distplot(_df['value'], kde=True, rug=False, bins=100, hist=True, color='darkblue', ax=ax)
                    _mean = _df['value'].mean()
                    _max = _df['value'].max()
                    _min = _df['value'].min()
                    _median = _df['value'].median()

                    ax.axvline(x=_mean, linewidth=1, color='green', ls='-.', lw=2)
                    ax.axvline(x=_median, linewidth=1, color='yellow', ls='--', lw=2)
                    ax.axvline(x=_max, linewidth=1, color='red', ls='-', lw=2)
                    ax.axvline(x=_min, linewidth=1, color='blue', ls='-', lw=2)

                    s = f"Min: {_min:.4f}\n"
                    s += f"Max: {_max:.4f}\n"
                    s += f"Avg: {_mean:.4f} \n"
                    s += f"Median: {_median:.4f}"
                    ax.text(x=0.6,
                            y=0.9,
                            s=s,
                            va='top',
                            bbox=dict(boxstyle='round4', edgecolor=(0, 0, 0, 1), fc=(1, 1, 1, 1)),
                            transform=ax.transAxes,
                            fontsize=18
                            )

                    ax.set_xlabel('Coeficient of Variation')
                    ax.set_ylabel('Frequency')

                    fig.suptitle(f'{item} {plot_title}')

                    plt.tight_layout()
                    plt.subplots_adjust(wspace=0.1, hspace=0.3)
                    plt.savefig(f"{Path(__file__).parent}/results/histogram.{prefix_title}_{item}.png")
                    if show:
                        plt.show()
                    plt.close()
        # ------------------------------------------------------------------------------------------------------------ #

        pbar = tqdm(range(4), desc=f'Charts: {self.dataset.name} {algorithm_name}')

        pbar.set_postfix(chart=f"plot_segmentation")
        if segmentation:
            plot_segments(scales=scales)
        pbar.update(1)

        sns.set(font_scale=1.3)
        pbar.set_postfix(chart=f"plot_cv_mean_roc")
        if mean_roc:
            plot_cv_mean_roc()
        pbar.update(1)

        sns.set(font_scale=1.3)
        pbar.set_postfix(chart=f"plot_combined_boxplot")
        if big_boxplot:
            plot_combined_boxplot()
        pbar.update(1)

        sns.set(font_scale=1.3)
        pbar.set_postfix(desc=f"plot_boxplots_histograms")
        if boxplots_histograms:
            plot_boxplots_histograms()
        pbar.update(1)
        # --------------------------------------------------------------------------------------------------------------- #

    # --------------------------------------------------------------------------------------------------------------- #

    def plot_segmentation(self, results: list):
        labels, trainingset = self.dataset.trainingset
        rgb = self.dataset.rgb_img_4display.copy()
        rgb[self.dataset.mask[:, :, :3]] = 0
        rgb[np.stack([np.where(trainingset > 0, True, False)]*3, 2)] = 0

        base_img = skimage.segmentation.mark_boundaries(rgb, trainingset, mode='outter', color=(1, 0, 0))
        colored_segs = skimage.color.label2rgb(trainingset,
                                               alpha=1,
                                               image_alpha=1,
                                               bg_label=0,
                                               colors=['yellow']
                                               )
        base_img = colored_segs + base_img

        fig, (axs) = plt.subplots(nrows=1, ncols=4, figsize=(30, 10))
        axs[0].imshow(base_img)
        axs[0].set_title('Reference polygons', fontsize=20)
        axs[0].axis("off")
        axs[0].grid('off')

        for i, (alg_name, param) in enumerate(results):

            if 'watershed' in alg_name:
                h5_key = f'{self.dataset.name}/{alg_name}/{param}/1e-05/segments'
                title = f'{alg_name} markers={param}'
            else:
                h5_key = f'{self.dataset.name}/{alg_name}/{param}/segments'
                title = f'{alg_name} k={param}'

            segmentation = self.results[h5_key][()]
            segmentation = skimage.morphology.closing(image=segmentation, selem=skimage.morphology.square(3))
            segmentation = skimage.measure.label(segmentation, connectivity=2)

            segmentation[self.dataset.mask[:, :, -1]] = 0
            boundaries = skimage.segmentation.mark_boundaries(rgb+colored_segs, segmentation, mode='outter', color=(1, 0, 0))
            # boundaries = skimage.segmentation.mark_boundaries(rgb, segmentation, mode='outter', color=(1, 0, 0))
            axs[i+1].imshow(boundaries)
            axs[i+1].set_title(title, fontsize=20)

            axs[i+1].axis("off")
            axs[i+1].grid('off')

        plt.subplots_adjust(wspace=0.005, hspace=0.01)
        plt.tight_layout()
        plt.savefig(f"{Path(__file__).parent}/results/final_segmentation_{self.dataset.name}_overlay.png")
        plt.show()
    # --------------------------------------------------------------------------------------------------------------- #

    def final_bar_graphs(self, items):
        df_list = []
        for item in items:
            df_list.append(self.post_segmentation_stats(*item))
        df = pd.concat(df_list)


        # if len(df.polygon_label_name.unique())<4:
        #     for i in range(4-len(df.polygon_label_name.unique())):
        #         for alg in ['kmeans','meanshift','watershed']:
        #             df = df.append([df.iloc[-1]], ignore_index=True)
        #             df['algorithm'].iloc[-1] = alg
        #             df['OS'].iloc[-1] = np.nan
        #             df['polygon_label_name'].iloc[-1] = f'extra_{i}'
        #             df['OS'].iloc[-1] = np.nan
        #             df['US'].iloc[-1] = np.nan
        #             df['ED'].iloc[-1] = np.nan
        #             df['IoU'].iloc[-1] = np.nan


        df = df[df.area_s >= 15]
        _filter = df.eval('(area_s >= 15) & ((area_intersection >= 0.5*area_r) or (area_intersection >= 0.5*area_s))')
        df['OS'][~(_filter)] = np.nan
        df['US'][~(_filter)] = np.nan
        df['ED'][~(_filter)] = np.nan
        df['IoU'][~(_filter)] = np.nan

        t = df[['algorithm', 'OS', 'US', 'ED','IoU']].groupby('algorithm').mean()
        t = t.reset_index().melt(id_vars=['algorithm'])
        if len(t) == 0:
            print("Error... no data to plot")
            return

        names_dict = {'urban1': 'Suburban',
                      'urban2': 'Urban',
                      'vegetation': 'Forest'
                      }

        sns.set(font_scale=2)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 7))

        sns.barplot(data=t, x='algorithm', y='value', hue='variable', ax=ax, dodge=True)
        def change_width(ax, new_value) :
            for patch in ax.patches :
                current_width = patch.get_width()
                patch.set_width(new_value)
                diff = current_width - new_value
                #patch.set_x(patch.get_x() - diff)
        change_width(ax, 0.2)


        ax.set_xlabel('')
        ax.set_title(names_dict[self.dataset.name])
        ax.set_ylim(0,1)
        handles, labels = ax.get_legend_handles_labels()
        ax.get_legend().remove()

        fig.legend(handles,
                   labels,
                   loc='upper right',
                   bbox_to_anchor=(0.95, 0.93, 0, 0),
                   bbox_transform=plt.gcf().transFigure,
                   fontsize=20
                   )

        plt.subplots_adjust(wspace=0.005, hspace=0.01)
        plt.tight_layout()
        plt.savefig(f"{Path(__file__).parent}/results/barchart_{names_dict[self.dataset.name]}.png")
        plt.show()
        plt.close()

        sns.set(font_scale=2)
        fig, axs = plt.subplots(nrows=1, ncols=len(items), figsize=(30, 8), sharey=False)
        for i, [algorithm_name, params] in enumerate(items):

            if algorithm_name == 'watershed':
                _title = f"markers={params}"
            elif algorithm_name == 'meanshift':
                _title = f"k seeds={params}"
            else:
                _title = f"k={params}"


            t = df[df.algorithm == algorithm_name]
            if len(t) == 0:
                print("Error... no data to plot")
                return

            t = t[['polygon_label_name', 'OS', 'US', 'ED','IoU']].groupby('polygon_label_name').mean()
            display(algorithm_name, t)
            t = t.reset_index().melt(id_vars='polygon_label_name')
            sns.barplot(data=t, x='polygon_label_name', hue='variable', y='value', ax=axs[i])
            #axs[i].set_title(f'{names_dict[self.dataset.name]} {algorithm_name} {_title}')
            axs[i].set_title(f'{algorithm_name} {_title}')
            axs[i].set_xlabel('')
            axs[i].set_ylabel('')
            axs[i].set_ylim(0,1)
            change_width(axs[i], .2)
            handles, labels = axs[i].get_legend_handles_labels()
            axs[i].get_legend().remove()

        fig.legend(handles, labels, loc='upper right',
                   bbox_to_anchor=(0.98, 0.87, 0, 0),
                   bbox_transform=plt.gcf().transFigure,
                   fontsize=15
                   )

        plt.subplots_adjust(wspace=0.1, hspace=0.01)
        plt.tight_layout()
        plt.savefig(f"{Path(__file__).parent}/results/barchart_{names_dict[self.dataset.name]} algorithms.png")
        plt.show()
        plt.close()