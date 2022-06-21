
Visual Descriptor Extraction
======
In this directoty we provide two datasets. The first one Ground Truth Corpus is a small one used for NLP purposes and the second one Patent Figure Dataset is used for CV purposes.

### 1. Ground Truth Corpus

This dataset contains figure captions for design patents from the USPTO database. Objects and Aspects are highlighted. The annotation follows a BIO schema. <br> 

This dataset is all about text. It can be used to train models in NLP domain.


### 2. Patent Figure Dataset
This dataset contains 66417 design patent figures along with their corresponding visual descriptors and metadata. <br>
Figures are in total 3G and they can be found in Google Drive link:
Figures are in PNG format. <br>

Visual descriptors and metadata are in a txt file which can be found in this derectory. This files gives the following infomation: <br>

*patentID*: This is the patent ID in the USPTO database. One patent has a unique ID                   <br>
*patentdate*: This is the data the patent was released.               <br>
*figid*: This is the index for figures within a patent. A patent may contain many figures.             <br>
*caption*: This is the figure caption.             <br>
*object*: What is the object in the figure             <br>
*aspect*: Which aspect of view is presented.             <br>
*figure_file*: This is the file name for a figure. It can be used to match figures in the dataset.              <br>
