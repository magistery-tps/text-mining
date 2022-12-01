# UBA - Maestria en Explotación de Datos y Descubrimiento de Conocimiento - Text Mining

El objetivo es predecir la rama del arbol de categorias, en al cual se encuentra un producto.

## Notebooks

* **Dataset Fashion Outfits**
   * [Pre-Processing](https://github.com/magistery-tps/text-mining/blob/master/notebooks/fashion-outfits/1.1-pre-processing.ipynb)
   * [EDA](https://github.com/magistery-tps/text-mining/blob/master/notebooks/fashion-outfits/2-eda.ipynb)
    * Models
      * [Baseline: Naive Bayes](https://github.com/magistery-tps/text-mining/blob/master/notebooks/fashion-outfits/3-naive-bayes-model.ipynb)
      * [BERT based classifier](https://github.com/magistery-tps/text-mining/blob/master/notebooks/fashion-outfits/4.1-bert-model.ipynb)
      * [W2V - FFNN](https://github.com/magistery-tps/text-mining/blob/master/notebooks/fashion-outfits/5_Farfetch_con_Keras_word2vect_prom.ipynb)
      * [SVD - Bayes - FFNN](https://github.com/magistery-tps/text-mining/blob/master/notebooks/fashion-outfits/6_ds_Farfetch_SVD_Bayes_RN.ipynb)
      * [Model Ensemple](https://github.com/magistery-tps/text-mining/blob/master/notebooks/fashion-outfits/7-ensamble.ipynb)
    * [Model Comparison Metrics](https://github.com/magistery-tps/text-mining/blob/master/notebooks/fashion-outfits/9-comparative-metrics.ipynb)

* **Dataset Ebay**
   * [Pre-Processing](https://github.com/magistery-tps/text-mining/blob/master/notebooks/ebay/1-pre-processing.ipynb)
   * [EDA](https://github.com/magistery-tps/text-mining/blob/master/notebooks/ebay/2-eda.ipynb)
   * [BERT based classifier](https://github.com/magistery-tps/text-mining/blob/master/notebooks/ebay/3-bert-model.ipynb)


## Requisitos

* [anaconda](https://www.anaconda.com/products/individual) / [miniconda](https://docs.conda.io/en/latest/miniconda.html) / [mamba (Recomendado)](https://github.com/mamba-org/mamba)
* [Setup de entorno (Window)](https://www.youtube.com/watch?v=O8YXuHNdIIk)



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

**Step 3**: Descargar imagenes del dataset fashion outfits.

```bash
$ cd datasets/fashion-outfits
$ wget https://storage.googleapis.com/sigir-challenge/images.tar.gz
$ tar -xvf images.tar.gz
```

**Nota**: las imagenes son requeridas para utilizar `FailReportGenerator`.


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


## Reporte de fallos

Este reporte contiene los ejemplos para los cuales el modelo fallo al predecir la categoria del producto. En el mismo se puede ver la descriptión del producto y las categorias predicha y real, junto con sus imagenes.
La idea de este reporte es comprender en que se equivoca el modelo. Para generar este reporte es necesario correr dos notebooks:
    
1. Notebook de pre-procesamiento: Esta notebook crea los conjuntos all, train, val, y test requieridos para generar el reporte.
2. Notebook bert-model: Por defecto TRAIN==False hay que activarlo para poder entrenar el modelo. En la etapa de evaluación se generadda el reporte.
    

## Datasets propuestos

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
