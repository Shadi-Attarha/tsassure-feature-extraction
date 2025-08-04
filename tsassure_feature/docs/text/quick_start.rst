.. _quick-start-label:

Quick Start
===========

Install YourPackage
-------------------

To install YourPackage, use pip:

.. code-block:: shell

   pip install tsassure-feature

If you need additional features, such as integration with Dask, install it with:

.. code-block:: shell

   pip install tsassure-feature[dask]

Dive In
-------

Before diving deep into the documentation, you can get started with this example:

We use two datasets named 'mainSet.xlsx' and 'NormalData.xlsx' which are available :ref:`here <main-set-link>` and :ref:`here <normal-data-link>`. The first column represents the sensor values for which we want to extract meaningful features. The dataset contains various sensor readings, and our goal is to extract meaningful features from them. Extracting meaningful features helps in better classifying data points and detecting abnormal measurements. Dataset 'NormalData.xlsx' is used to find other correlated time-series data to the main sensor (first column), and it is part of dataset :ref:`[1]` where only normal data is available.

To extract the features for dataset 'mainSet.xlsx', we need to write:

The `mainSet` DataFrame has the following structure:

.. code-block:: python

   +-------------+------------+----------+----------+
   | Merge_Humi  | Merge_Temp | Humi_Wea | Temp_Wea |
   +=============+============+==========+==========+
   | 54.25       | 24.275     | 29       | 30.0024  |
   +-------------+------------+----------+----------+
   | 58          | 23.9499    | 29       | 30.0024  |
   +-------------+------------+----------+----------+
   | ...         | ...        | ...      | ...      |
   +-------------+------------+----------+----------+

The first column is the main column for which we intend to extract meaningful features. All columns in this dataframe are different sensors' readings represented in the columns `Merge_Humi`, `Merge_Temp`, `Humi_Wea`, `Temp_Wea`.

To extract features from these time series:

.. code-block:: python

   from tsassure_feature.feature_extractor import FeatureExtractor
   import pandas as pd

   fe = FeatureExtractor("NormalData.xlsx", "mainSet.xlsx")
   df, correlateddf = fe.extract_features()

You will get a DataFrame `df` with meaningful features. Print to see the df and then you can save it in a new Excel file:

.. code-block:: python

   df.head()
   excel_file = 'Output.xlsx'
   # Save the DataFrame to Excel
   df.to_excel(excel_file, index=False)

%For a complete example, see the Jupyter notebook `TsAssure.ipynb <https://colab.research.google.com/drive/1tHabIjiNofVFtG9WGYUxxYbJQ6fKT3gX?usp=drive_link>`_.

References
==========

.. _main-set-link: https://github.com/shadi-attarha/tsassure-feature-extraction/blob/main/mainSet.xlsx

.. _normal-data-link: https://github.com/shadi-attarha/tsassure-feature-extraction/blob/main/NormalData.xlsx

.. [1] `Main Set <https://github.com/shadi-attarha/tsassure-feature-extraction/blob/main/mainSet.xlsx>`_
.. [2] `Normal Data <https://github.com/shadi-attarha/tsassure-feature-extraction/blob/main/NormalData.xlsx>`_
