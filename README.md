{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "IZ6SNYq_tVVC"
   },
   "source": [
    "# Classify text with BERT\n",
    "\n",
    "### Learning Objectives\n",
    "1. Learn how to load a pre-trained BERT model from TensorFlow Hub\n",
    "2. Learn how to build your own model by combining with a classifier\n",
    "3. Learn how to train a BERT model by fine-tuning\n",
    "4. Learn how to save your trained model and use it\n",
    "5. Learn how to evaluate a text classification model\n",
    "\n",
    "This lab will show you how to fine-tune BERT to perform sentiment analysis on a dataset of plain-text IMDB movie reviews.\n",
    "In addition to training a model, you will learn how to preprocess text into an appropriate format.\n",
    "\n",
    "### Before you start\n",
    "\n",
    "Please ensure you have a GPU (1 x NVIDIA Tesla T4 should be enough) attached to your Notebook instance to ensure that the training doesn't take too long. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "2PHBpLPuQdmK"
   },
   "source": [
    "## About BERT\n",
    "\n",
    "[BERT](https://arxiv.org/abs/1810.04805) and other Transformer encoder architectures have been wildly successful on a variety of tasks in NLP (natural language processing). They compute vector-space representations of natural language that are suitable for use in deep learning models. The BERT family of models uses the Transformer encoder architecture to process each token of input text in the full context of all tokens before and after, hence the name: Bidirectional Encoder Representations from Transformers. \n",
    "\n",
    "BERT models are usually pre-trained on a large corpus of text, then fine-tuned for specific tasks.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "import os\n",
    "import warnings\n",
    "\n",
    "warnings.filterwarnings(\"ignore\")\n",
    "os.environ[\"TF_CPP_MIN_LOG_LEVEL\"] = \"2\"\n",
    "\n",
    "# Set `PATH` to include the directory containing saved_model_cli\n",
    "PATH = %env PATH\n",
    "%env PATH=/home/jupyter/.local/bin:{PATH}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "_XgTpm9ZxoN9",
    "tags": []
   },
   "outputs": [],
   "source": [
    "import datetime\n",
    "import shutil\n",
    "\n",
    "import matplotlib.pyplot as plt\n",
    "import tensorflow as tf\n",
    "import tensorflow_hub as hub\n",
    "import tensorflow_text as text\n",
    "from google.cloud import aiplatform\n",
    "from official.nlp import optimization  # to create AdamW optmizer\n",
    "\n",
    "tf.get_logger().setLevel(\"ERROR\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "To check if you have a GPU attached. Run the following."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(\"Num GPUs Available: \", len(tf.config.list_physical_devices(\"GPU\")))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "q6MugfEgDRpY"
   },
   "source": [
    "### Sentiment Analysis\n",
    "\n",
    "This notebook trains a sentiment analysis model to classify movie reviews as *positive* or *negative*, based on the text of the review.\n",
    "\n",
    "You'll use the [Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/) that contains the text of 50,000 movie reviews from the [Internet Movie Database](https://www.imdb.com/)."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "Vnvd4mrtPHHV"
   },
   "source": [
    "### Download the IMDB dataset\n",
    "\n",
    "Let's download and extract the dataset, then explore the directory structure.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "pOdqCMoQDRJL",
    "tags": []
   },
   "outputs": [],
   "source": [
    "url = \"https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz\"\n",
    "\n",
    "# Set a path to a folder outside the git repo. This is important so data won't get indexed by git on Jupyter lab\n",
    "path = \"/home/jupyter/\"\n",
    "\n",
    "dataset = tf.keras.utils.get_file(\n",
    "    \"aclImdb_v1.tar.gz\", url, untar=True, cache_dir=path, cache_subdir=\"\"\n",
    ")\n",
    "\n",
    "dataset_dir = os.path.join(os.path.dirname(dataset), \"aclImdb\")\n",
    "\n",
    "train_dir = os.path.join(dataset_dir, \"train\")\n",
    "\n",
    "# remove unused folders to make it easier to load the data\n",
    "remove_dir = os.path.join(train_dir, \"unsup\")\n",
    "shutil.rmtree(remove_dir)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "lN9lWCYfPo7b"
   },
   "source": [
    "Next, you will use the `text_dataset_from_directory` utility to create a labeled `tf.data.Dataset`.\n",
    "\n",
    "The IMDB dataset has already been divided into train and test, but it lacks a validation set. Let's create a validation set using an 80:20 split of the training data by using the `validation_split` argument below.\n",
    "\n",
    "Note:  When using the `validation_split` and `subset` arguments, make sure to either specify a random seed, or to pass `shuffle=False`, so that the validation and training splits have no overlap."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "6IwI_2bcIeX8",
    "tags": []
   },
   "outputs": [],
   "source": [
    "AUTOTUNE = tf.data.AUTOTUNE\n",
    "batch_size = 32\n",
    "seed = 42\n",
    "\n",
    "raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(\n",
    "    path + \"aclImdb/train\",\n",
    "    batch_size=batch_size,\n",
    "    validation_split=0.2,\n",
    "    subset=\"training\",\n",
    "    seed=seed,\n",
    ")\n",
    "\n",
    "class_names = raw_train_ds.class_names\n",
    "train_ds = raw_train_ds.cache().prefetch(buffer_size=AUTOTUNE)\n",
    "\n",
    "val_ds = tf.keras.preprocessing.text_dataset_from_directory(\n",
    "    path + \"aclImdb/train\",\n",
    "    batch_size=batch_size,\n",
    "    validation_split=0.2,\n",
    "    subset=\"validation\",\n",
    "    seed=seed,\n",
    ")\n",
    "\n",
    "val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)\n",
    "\n",
    "test_ds = tf.keras.preprocessing.text_dataset_from_directory(\n",
    "    path + \"aclImdb/test\", batch_size=batch_size\n",
    ")\n",
    "\n",
    "test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "HGm10A5HRGXp"
   },
   "source": [
    "Let's take a look at a few reviews."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "JuxDkcvVIoev",
    "tags": []
   },
   "outputs": [],
   "source": [
    "for text_batch, label_batch in train_ds.take(1):\n",
    "    for i in range(3):\n",
    "        print(f\"Review: {text_batch.numpy()[i]}\")\n",
    "        label = label_batch.numpy()[i]\n",
    "        print(f\"Label : {label} ({class_names[label]})\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "dX8FtlpGJRE6"
   },
   "source": [
    "## Loading models from TensorFlow Hub\n",
    "\n",
    "For the purpose of this lab, we will be loading a model called Small BERT. Small BERT has the same general architecture as the original BERT but the has fewer and/or smaller Transformer blocks. \n",
    "\n",
    "Some other popular BERT models are BERT Base, ALBERT, BERT Experts, Electra. See the continued learning section at the end of this lab for more info. \n",
    "\n",
    "Aside from the models available below, there are [multiple versions](https://tfhub.dev/google/collections/transformer_encoders_text/1) of the models that are larger and can yeld even better accuracy but they are too big to be fine-tuned on a single GPU. You will be able to do that on the [Solve GLUE tasks using BERT on a TPU colab](https://www.tensorflow.org/tutorials/text/solve_glue_tasks_using_bert_on_tpu).\n",
    "\n",
    "You'll see in the code below that switching the tfhub.dev URL is enough to try any of these models, because all the differences between them are encapsulated in the SavedModels from TF Hub."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Choose a BERT model to fine-tune"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "cellView": "form",
    "id": "y8_ctG55-uTX",
    "tags": []
   },
   "outputs": [],
   "source": [
    "# defining the URL of the smallBERT model to use\n",
    "tfhub_handle_encoder = (\n",
    "    \"https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1\"\n",
    ")\n",
    "\n",
    "# defining the corresponding preprocessing model for the BERT model above\n",
    "tfhub_handle_preprocess = (\n",
    "    \"https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3\"\n",
    ")\n",
    "\n",
    "print(f\"BERT model selected           : {tfhub_handle_encoder}\")\n",
    "print(f\"Preprocess model auto-selected: {tfhub_handle_preprocess}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "7WrcxxTRDdHi"
   },
   "source": [
    "## Preprocessing model\n",
    "\n",
    "Text inputs need to be transformed to numeric token ids and arranged in several Tensors before being input to BERT. TensorFlow Hub provides a matching preprocessing model for each of the BERT models discussed above, which implements this transformation using TF ops from the TF.text library. It is not necessary to run pure Python code outside your TensorFlow model to preprocess text.\n",
    "\n",
    "The preprocessing model must be the one referenced by the documentation of the BERT model, which you can read at the URL printed above. For BERT models from the drop-down above, the preprocessing model is selected automatically.\n",
    "\n",
    "Note: You will load the preprocessing model into a [hub.KerasLayer](https://www.tensorflow.org/hub/api_docs/python/hub/KerasLayer) to compose your fine-tuned model. This is the preferred API to load a TF2-style SavedModel from TF Hub into a Keras model."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Use `hub.KerasLayer` to initialize the preprocessing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "bert_preprocess_model = hub.KerasLayer(tfhub_handle_preprocess)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "x4naBiEE_cZX"
   },
   "source": [
    "Let's try the preprocessing model on some text and see the output:"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Call the preprocess model function and pass text_test"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "r9-zCzJpnuwS",
    "tags": []
   },
   "outputs": [],
   "source": [
    "text_test = [\"this is such an amazing movie!\"]\n",
    "text_preprocessed = bert_preprocess_model(text_test)\n",
    "\n",
    "# This print box will help you inspect the keys in the pre-processed dictionary\n",
    "print(f\"Keys       : {list(text_preprocessed.keys())}\")\n",
    "\n",
    "# 1. input_word_ids is the ids for the words in the tokenized sentence\n",
    "print(f'Shape      : {text_preprocessed[\"input_word_ids\"].shape}')\n",
    "print(f'Word Ids   : {text_preprocessed[\"input_word_ids\"][0, :12]}')\n",
    "\n",
    "# 2. input_mask is the tokens which we are masking (masked language model)\n",
    "print(f'Input Mask : {text_preprocessed[\"input_mask\"][0, :12]}')\n",
    "\n",
    "# 3. input_type_ids is the sentence id of the input sentence.\n",
    "print(f'Type Ids   : {text_preprocessed[\"input_type_ids\"][0, :12]}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "EqL7ihkN_862"
   },
   "source": [
    "As you can see, now you have the 3 outputs from the preprocessing that a BERT model would use (`input_words_id`, `input_mask` and `input_type_ids`).\n",
    "\n",
    "Some other important points:\n",
    "- The input is truncated to 128 tokens. \n",
    "- The `input_type_ids` only have one value (0) because this is a single sentence input. For a multiple sentence input, it would have one number for each input.\n",
    "\n",
    "The text pre-processor is a TensorFlow model. This means that instead of pre-processing separately, we can include it as a layer in the model code."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "DKnLPSEmtp9i"
   },
   "source": [
    "### Using the BERT model\n",
    "\n",
    "Before putting BERT into your own model, let's take a look at its outputs. You will load it from TF Hub and see the returned values."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "tXxYpK8ixL34",
    "tags": []
   },
   "outputs": [],
   "source": [
    "bert_model = hub.KerasLayer(tfhub_handle_encoder)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "_OoF9mebuSZc",
    "tags": []
   },
   "outputs": [],
   "source": [
    "bert_results = bert_model(text_preprocessed)\n",
    "\n",
    "print(f\"Loaded BERT: {tfhub_handle_encoder}\")\n",
    "print(f'Pooled Outputs Shape:{bert_results[\"pooled_output\"].shape}')\n",
    "print(f'Pooled Outputs Values:{bert_results[\"pooled_output\"][0, :12]}')\n",
    "print(f'Sequence Outputs Shape:{bert_results[\"sequence_output\"].shape}')\n",
    "print(f'Sequence Outputs Values:{bert_results[\"sequence_output\"][0, :12]}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "sm61jDrezAll"
   },
   "source": [
    "The BERT models return a map with 3 important keys: `pooled_output`, `sequence_output`, `encoder_outputs`:\n",
    "\n",
    "- `pooled_output` to represent each input sequence as a whole. The shape is `[batch_size, H]`. You can think of this as an embedding for the entire movie review.\n",
    "- `sequence_output` represents each input token in the context. The shape is `[batch_size, seq_length, H]`. You can think of this as a contextual embedding for every token in the movie review.\n",
    "- `encoder_outputs` are the intermediate activations of the `L` Transformer blocks. `outputs[\"encoder_outputs\"][i]` is a Tensor of shape `[batch_size, seq_length, 1024]` with the outputs of the i-th Transformer block, for `0 <= i < L`. The last value of the list is equal to `sequence_output`.\n",
    "\n",
    "For the fine-tuning you are going to use the `pooled_output` array."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "pDNKfAXbDnJH"
   },
   "source": [
    "### Define your model\n",
    "\n",
    "You will create a very simple fine-tuned model, with the preprocessing model, the selected BERT model, one Dense and a Dropout layer.\n",
    "\n",
    "Note: for more information about the base model's input and output you can use copy the model's url to get to the documentation page."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The order of the layers in the model will be:\n",
    "1. Input Layer\n",
    "2. Pre-processing Layer\n",
    "3. Encoder Layer\n",
    "4. From the BERT output map, use pooled_output\n",
    "5. Dropout layer\n",
    "6. Dense layer with sigmoid activation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "aksj743St9ga",
    "tags": []
   },
   "outputs": [],
   "source": [
    "def build_classifier_model(dropout_rate=0.1):\n",
    "    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name=\"text\")\n",
    "    preprocessing_layer = hub.KerasLayer(\n",
    "        tfhub_handle_preprocess, name=\"preprocessing\"\n",
    "    )\n",
    "    encoder_inputs = preprocessing_layer(text_input)\n",
    "    encoder = hub.KerasLayer(\n",
    "        tfhub_handle_encoder, trainable=True, name=\"BERT_encoder\"\n",
    "    )\n",
    "    outputs = encoder(encoder_inputs)\n",
    "    net = outputs[\"pooled_output\"]\n",
    "    net = tf.keras.layers.Dropout(dropout_rate)(net)\n",
    "    net = tf.keras.layers.Dense(1, activation=\"sigmoid\", name=\"classifier\")(net)\n",
    "    return tf.keras.Model(text_input, net)\n",
    "\n",
    "\n",
    "# Let's check that the model runs with the output of the preprocessing model.\n",
    "dropout_rate = 0.15\n",
    "classifier_model = build_classifier_model(dropout_rate)\n",
    "bert_raw_result = classifier_model(tf.constant(text_test))\n",
    "print(bert_raw_result)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "ZTUzNV2JE2G3"
   },
   "source": [
    "The output is meaningless, of course, because the model has not been trained yet.\n",
    "\n",
    "Let's take a look at the model's structure."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "0EmzyHZXKIpm",
    "tags": []
   },
   "outputs": [],
   "source": [
    "tf.keras.utils.plot_model(classifier_model)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "WbUWoZMwc302"
   },
   "source": [
    "## Model training\n",
    "\n",
    "You now have all the pieces to train a model, including the preprocessing module, BERT encoder, data, and classifier."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "WpJ3xcwDT56v"
   },
   "source": [
    "### Loss function\n",
    "\n",
    "Since this is a binary classification problem and the model outputs a probability (a single-unit layer), you'll use `losses.BinaryCrossentropy` loss function.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Define your loss and evaluation metrics here. Since it is a binary classification use BinaryCrossentropy and BinaryAccuracy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "OWPOZE-L3AgE",
    "tags": []
   },
   "outputs": [],
   "source": [
    "loss = tf.keras.losses.BinaryCrossentropy()\n",
    "metrics = tf.metrics.BinaryAccuracy()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "77psrpfzbxtp"
   },
   "source": [
    "### Optimizer\n",
    "\n",
    "For fine-tuning, let's use the same optimizer that BERT was originally trained with: the \"Adaptive Moments\" (Adam). This optimizer minimizes the prediction loss and does regularization by weight decay (not using moments), which is also known as [AdamW](https://www.tensorflow.org/addons/api_docs/python/tfa/optimizers/AdamW).\n",
    "\n",
    "In past labs, we have been using the Adam optimizer which is a popular choice. However, for this lab we will be using a new optimizier which is meant to improve generalization. The intuition and algoritm behind AdamW can be found in this paper [here](https://arxiv.org/abs/1711.05101).\n",
    "\n",
    "For the learning rate (`init_lr`), we use the same schedule as BERT pre-training: linear decay of a notional initial learning rate, prefixed with a linear warm-up phase over the first 10% of training steps (`num_warmup_steps`). In line with the BERT paper, the initial learning rate is smaller for fine-tuning (best of 5e-5, 3e-5, 2e-5)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "P9eP2y9dbw32",
    "tags": []
   },
   "outputs": [],
   "source": [
    "epochs = 5\n",
    "steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()\n",
    "num_train_steps = steps_per_epoch * epochs\n",
    "num_warmup_steps = int(0.1 * num_train_steps)\n",
    "\n",
    "init_lr = 3e-5\n",
    "optimizer = optimization.create_optimizer(\n",
    "    init_lr=init_lr,\n",
    "    num_train_steps=num_train_steps,\n",
    "    num_warmup_steps=num_warmup_steps,\n",
    "    optimizer_type=\"ada
