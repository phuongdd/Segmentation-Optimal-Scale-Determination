"""."""

import glob
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.signal
import scipy.ndimage
import scipy.stats

import geopandas as gpd
import rasterio
from rasterio import features

import skimage.util
import sklearn.cluster
import sklearn.preprocessing

import skimage
import skimage.color
import skimage.exposure
import skimage.segmentation
import skimage.filters
import skimage.morphology

import property_manager


class HSIDataset:
    """ Provide a unified interface for all Hyperspectral datasets """

    def __init__(self, file_path, name):
        self.path = Path(file_path)
        self.file_name = self.path.name
        self.base_name = Path(self.path).stem
        self.base_dir = Path(self.path).parent

        self.train_path = self.base_dir / "trainingset"
        self.test_path = self.base_dir / "testset"
        self.name = name
        self.rgb_gain = 5

    @property_manager.cached_property
    def raster(self):
        return rasterio.open(str(self.path))

    @property
    def original_n_bands(self):
        return self.raster.count

    @property_manager.cached_property
    def hsi_bands(self):
        return np.arange(400, 902, 2)

    @property
    def n_bands(self):
        return len(self.hsi_bands)

    @property
    def n_cols(self):
        return self.raster.width

    @property
    def n_rows(self):
        return self.raster.height

    @property_manager.cached_property
    def n_pixels(self):
        return self.n_rows * self.n_cols

    @property_manager.cached_property
    def rgb_bands(self):
        # RGB bands  defined by Phuong: [230, 136, 75]
        # red, green, blue = [230, 136, 75]
        # NIR=850, Red = 670, Green = 540, Blue = 470

        red = np.where(self.hsi_bands == 670)[0][0]
        green = np.where(self.hsi_bands == 540)[0][0]
        blue = np.where(self.hsi_bands == 470)[0][0]

        return [red, green, blue]

    @property_manager.cached_property
    def mask(self):
        return self.hsi.mask

    @property_manager.cached_property
    def hsi(self):
        # Move the bands dimension to the third one (more intuitive as the "z" band)
        # Also cut up to the 251 bands (self.n_bands)
        raw_array = np.moveaxis(self.raster.read(range(1, self.n_bands+1)), 0, -1)
        raw_array = raw_array.astype(np.float32)
        mask = np.where(raw_array == 0, False, True)
        raw_array = raw_array/10000

        # Smooth the noisy signal
        noise_removed = scipy.signal.savgol_filter(raw_array, window_length=25, polyorder=3, mode='interp', axis=2)
        hsi_arr = np.ma.masked_where(np.logical_not(mask), noise_removed)
        return hsi_arr

    @property_manager.cached_property
    def rgb(self):
        return np.ascontiguousarray(self.hsi[:, :, self.rgb_bands])

    @property_manager.cached_property
    def rgb_img_4display(self):
        return skimage.exposure.adjust_gamma(self.rgb, gain=self.rgb_gain)

    @property_manager.cached_property
    def feature_vectors(self):
        x = np.arange(self.n_cols)
        y = np.arange(self.n_rows)
        xx, yy = np.meshgrid(x, y)
        xx = xx.reshape(self.n_rows, self.n_cols, 1) / (self.n_cols/10)
        yy = yy.reshape(self.n_rows, self.n_cols, 1) / (self.n_rows/10)

        rgb_xy = np.dstack([self.rgb, xx, yy])
        hsi_xy = np.dstack([self.hsi, xx, yy])
        pca_xy = np.dstack([self.pca, xx, yy])

        feature_vectors = {}
        feature_vectors['HSI_XY'] = hsi_xy.astype(np.float32)
        feature_vectors['RGB_XY'] = rgb_xy.astype(np.float32)
        feature_vectors['PCA_XY'] = pca_xy.astype(np.float32)

        feature_vectors['HSI'] = self.hsi.astype(np.float32)
        feature_vectors['RGB'] = self.rgb.astype(np.float32)
        feature_vectors['PCA'] = self.pca.astype(np.float32)

        return feature_vectors

    @property_manager.cached_property
    def trainingset(self):
        return self.__vector_to_raster(role='train')

    @property_manager.cached_property
    def testset(self):
        return self.__vector_to_raster(role='test')

    def __vector_to_raster(self, role='train'):
        if role == 'test':
            path = str(Path(self.test_path)/'*.shp')
        else:
            path = str(Path(self.train_path)/'*.shp')

        out = np.zeros(shape=(self.n_rows, self.n_cols), dtype=np.uint8)

        trainingset = []
        categories = []
        shapes = []
        files = glob.glob(pathname=path)
        for file in files:
            shapes.append(gpd.read_file(file))
        if len(shapes) > 0:
            shapes = pd.concat(shapes)
            categories = ['undefined']+list(shapes.Label.unique())
            # Start from 1, because background will be zero
            categories = dict(zip(categories, range(0, len(categories)+1)))
            shapes['label_id'] = shapes.Label.apply(lambda s: categories[s])
            shapes = [(geom, label_id) for geom, label_id
                      in zip(shapes.geometry, shapes.label_id)
                      if geom is not None
                      ]

            trainingset = features.rasterize(shapes=shapes, fill=0, out=out, transform=self.raster.transform)

        return categories, np.array(trainingset)
