import numpy as np
from hsi_dataset import HSIDataset
from hsi_segmentation import HSISegmentation

tiff_files = [['/Users/kiran/PhD/hsi_segmentation/data/forest/20170820_Forest_Final_INT.tif', 'vegetation'],
              ['/Users/kiran/PhD/hsi_segmentation/data/urban2/20170820_Urban2_INT_Final.tif', 'urban2'],
              ['/Users/kiran/PhD/hsi_segmentation/data/urban1/20170820_Urban_Ref_Reg_Subset.tif', 'urban1']
              ]

runs = [  # ['meanshift',  {'k': np.arange(0, 550, 10)}],
    ['kmeans',     {'k': np.arange(0, 550, 10)}],
    #['watershed', {'markers': np.arange(0, 10100, 100), 'compactness': [1e-5]}]
]


chart_args = dict(mean_roc=True,
                  big_boxplot=True,
                  boxplots_histograms=True,
                  show=False,
                  force_stats=True,
                  )

for tiff_file, name in tiff_files:
    dataset = HSIDataset(file_path=tiff_file, name=name)
    seg = HSISegmentation(dataset=dataset)

    for algorithm_name, hparams in runs:
        # print(f'\nSegmentation: {name} {algorithm_name}')
        # ret = seg.run(algorithm_name=algorithm_name, hparams=hparams, force=False)

        # print(f'\nStats: {name} {algorithm_name} ')
        # _, _ = seg.stats(algorithm_name=algorithm_name, stat_name='coef_variation', force=False, fix_small=True)

        print(f'\nCharts: {name} {algorithm_name}')
        seg.plot_charts(algorithm_name, **chart_args)

        print('')
        print('='*80)
        print('')
