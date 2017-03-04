## Running the Udacity Deep Learning Nanodegree Foundation Image Classification Project on Floydhub.com

Original Repository: [LudwikTrammer / deep-learning](https://github.com/ludwiktrammer/deep-learning)

1. Create an account on [Floydhub.com](https://www.floydhub.com) (don't forget
to confirm your email). You will automatically receive 100 free GPU hours.

1. Install the `floyd` command on your computer:

        pip install -U floyd-cli

    Do this even if you already installed `floyd-cli` before, to ensure you have
    the most recent version.

1. Authenticate using your Floydhub credentials:

        floyd login

    The Floydhub welcome page will be launched by your default web browser
    with an authentication token displayed near the bottom.  Copy and paste the
    token into the prompt in your terminal.

1. Clone this repository:

        git clone https://github.com/frankhinek/Deep-Learning-Examples.git

    Note: There are a few minor differences between this repository and the
    original Udacity repository, which are detailed later in this README.

1. Enter the folder for the image classification project:

        cd image-classification

1. Initialize a Floydhub project:

        floyd init dlnd_image_classification

1. Run the project:

        floyd run --gpu --env tensorflow --mode jupyter --data diSgciLH4WA7HpcHNasP9j

    It will be run on a machine with GPU (`--gpu`), using a Tensorflow 0.12.1 +
    Keras 1.2.2 on Python3 environment (`--env tensorflow`), as a Jupyter
    notebook (`--mode jupyter`), with Floyd's built-in cifar-10 dataset
    available (`--data diSgciLH4WA7HpcHNasP9j`).

1. Wait for the Jupyter notebook to become available and then access the URL
displayed in the terminal (described as "Path to jupyter notebook").

1. Remember to explicitly stop the experiment when you are not using the
notebook. As long as it runs (even in the background) it will cost GPU hours.
You can stop an experiment in the ["Experiments" section on floyd.com](https://www.floydhub.com/experiments)
or using the `floyd stop` command:

        floyd stop ID

    where `ID` is the "RUN ID" displayed in the terminal when you run the
    project.  You can list the RUN IDs of recent projects using the
    `floyd status` command.

**Important:** When you run a project it will always start from scratch
(i.e. from the state present *locally* on your computer). If you made changes in
the remote jupyter notebook during a previous run, the changes will **not** be
present in subsequent runs. To make them permanent you need to add the changes
to your local project directory. When running the notebook you can download them
directly from Jupyter - *File / Download / Notebook*. After downloading the
notebook, replace your local `dlnd_image_classification.ipynb` file with the
version you downloaded.

Alternatively, if you already stoped the experiment, you can still download the
file using the `floyd output` command:

        floyd output ID

    where `ID` is the "RUN ID" displayed in the terminal when you run the
    project.  You can list the RUN IDs of recent projects using the
    `floyd status` command.

Just run the command above, download `dlnd_image_classification.ipynb` and
replace your local version with the newly downloaded one.

## How is this repository different from [the original](https://github.com/udacity/deep-learning)?

1. [Ludwik Trammer](https://github.com/ludwiktrammer) added support for
Floydhub's built-in CIFAR-10 dataset. If its presence is detected, it will be
used, without a need to download anything. You can [read more about datasets provided by Floydhub](http://docs.floydhub.com/guides/datasets/)).

2. [Ludwik Trammer](https://github.com/ludwiktrammer) added a
`floyd_requirements.txt` file, so that the tqdm package dependency Udacity used
in the predefined code is installed into the Floydhub run time environment. You
can [about how Floydhub handles installing additional Python packages](http://docs.floydhub.com/home/installing_dependencies/).

3. [Ludwik Trammer](https://github.com/ludwiktrammer) added a `.floydignore`
file to stop the 1.5GiB of CIFAR-10 data from being uploaded to Floyd.  This
wastes time, can cause timeouts, and will result in data charges being assessed.
There is limited documentation at the moment, but you can [read about `.floydignore` files on the Floydhub documentation site](http://docs.floydhub.com/commands/init/#description).

3. [Ludwik Trammer](https://github.com/ludwiktrammer) added this README, which I
subsequently modified.
