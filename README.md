# Clone this repo (or copy the files)
git clone git@github.com:kiranmantri/hsi_segmentation.git


# Put all the data in data folder.
It should like like this:

```
.
├── LICENSE
├── README.md
├── data
│   ├── forest
│   ├── test_image
│   ├── urban1
│   ├── urban2
│   └── urban_full
├── models
│   └── 20170820_Urban_Ref_Reg_Subset.rf_classifier.h5
├── hsi_dataset.py
├── hsi_segmentation.py
├── post-segmentation-stats.ipynb
├── process_results.ipynb
├── requirements.txt
├── run_segmentation.ipynb
└── run_segmentation_stats_charts.py

```

# Required extensions Jupyter Notebook
```bash
jupyter nbextension install @jupyterlab/plotly-extension
jupyter nbextension install @jupyter-widgets/jupyterlab-manager\n
jupyter nbextension enable --py widgetsnbextension
```



# How to use it:
Create a Notebook

```
import sys
sys.path.append('<folder where you put the hsi_segmentation files>')

from hsi_segmentation import HSISegmentation
from hsi_dataset import HSIDataset


dataset = HSIDataset(file_path='<tiff file>')
seg = HSISegmentation(dataset=dataset)


clf = seg.random_forest(mode="train")
results, confusion_matrix = seg.random_forest(mode="test")

predicted = seg.random_forest(mode="predict")
plt.imshow(predicted)


```
