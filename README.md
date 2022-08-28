# UBA - Maestria en Explotación de Datos y Descubrimiento de Conocimiento - Text Mining


## Notebooks

* Ebay: Predicción de categoria de productos
   * [Pre-Processing](https://github.com/magistery-tps/text-mining/blob/master/notebooks/ebay-prepprocessing.ipynb)
   * [Clusters](https://github.com/magistery-tps/text-mining/blob/master/notebooks/ebay-clusters.ipynb)
   * [Model](https://github.com/magistery-tps/text-mining/blob/master/notebooks/ebay-model.ipynb)


## Datasets

### Clasificacion de productos

En todos los casos la tareas consta de predecir la categoria de un producto en base a su título.

* [UK- e-bay](https://data.world/opensnippets/ebay-uk-products-dataset)
* MeLi Data Challenge 2019
    * [Kaggle](https://www.kaggle.com/datasets/abugim/meli-data-challenge-2019)
    * [Train Set](https://meli-data-challenge.s3.amazonaws.com/train.csv.gz)
    * [Test Set](https://meli-data-challenge.s3.amazonaws.com/test.csv)
    * [Submit sample](https://meli-data-challenge.s3.amazonaws.com/sample_submission.csv)
    * [MeLi Data Challenge 2019 – Multiclass Classification in Keras](https://eduardofv.com/2019/10/04/meli-data-challenge-2019-multiclass-classification-in-keras/)
    * [MeLi Data Challenge 2019 | Deep Learning](https://github.com/mlacosta/MeLi-Data-Challenge-2019)
* [MeLi Data Challenge 2020](https://www.kaggle.com/datasets/marlesson/meli-data-challenge-2020)
* Amazon
    * [Amazon Product Dataset 2020](https://www.kaggle.com/datasets/promptcloud/amazon-product-dataset-2020)
    * [10,000_Amazon_Products_Dataset](https://www.kaggle.com/datasets/nguyenngocphung/10000-amazon-products-dataset)
    * [Toy Products on Amazon](https://www.kaggle.com/datasets/PromptCloudHQ/toy-products-on-amazon)
    * [Amazon Product Reviews Dataset](https://www.kaggle.com/datasets/promptcloud/amazon-product-reviews-dataset)
* [Google Play Store Apps](https://www.kaggle.com/datasets/lava18/google-play-store-apps)
 
### Stock prices prediction

* [Two Sigma: Using News to Predict Stock Movements](https://www.kaggle.com/competitions/two-sigma-financial-news/rules)


## Requisitos

* [anaconda](https://www.anaconda.com/products/individual) / [miniconda](https://docs.conda.io/en/latest/miniconda.html) / [mamba (Recomendado)](https://github.com/mamba-org/mamba)
* [Setup de entorno (Window](https://www.youtube.com/watch?v=O8YXuHNdIIk)

## Comenzando


**Step 1**: Clonar repo.

```bash
$ git clone https://github.com/adrianmarino/text-mining.git
$ cd text-mining
```

**Step 2**: Crear environment.

```bash
$ conda env create -f environment.yml
```

## Ver notebooks en jupyter lab

**Step 1**: Activar environment.

```bash
$ conda activate text-mining
```

**Step 2**: Sobre el directorio del proyecto ejecutar `jupyter lab`.

```bash
$ jupyter lab

Jupyter Notebook 6.1.4 is running at:
http://localhost:8888/?token=45efe99607fa6......
```

**Step 3**: Ir a http://localhost:8888.... como se indica en la consola.
