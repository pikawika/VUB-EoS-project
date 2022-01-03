# Install instructions for Ubuntu

This project makes use of Python 3 and typical Python packages. We use a Anaconda environment for this. The Ubuntu install instructions are given below. An exported environment is also available.

## Table of contents

- [Author](#author)
- [Setting up Anaconda environment](#setting-up-anaconda-environment)
  * [Configuring the base environment](#configuring-the-base-environment)
  * [Configuring the environment from the YML file](#configuring-the-environment-from-the-yml-file)

<hr>


## Author

| Name             | Role                 | VUB mail                                                  | Personal mail                                               |
| ---------------- | -------------------- | --------------------------------------------------------- | ----------------------------------------------------------- |
| Lennert Bontinck | Master Thesis writer | [lennert.bontinck@vub.be](mailto:lennert.bontinck@vub.be) | [info@lennertbontinck.com](mailto:info@lennertbontinck.com) |

<hr>

## Setting up Anaconda environment

The instructions below highlight the steps needed to recreate the used anaconda environment. It's also possible to import the environment used.

### Configuring the base environment

- Install [the free version of Anaconda Navigator](https://www.anaconda.com/products/individual).

- Launch the Anaconda Navigator GUI using `anaconda-navigator`.

- From the Anaconda Navigator GUI, create a new environment named `eos-project`.

  - Include both Python and R. The following versions were used:
    - `Python 3.8.12`
    -  `R 3.6.1`
  - Doing so should install a whole suite of packages by default 

- Using your terminal, activate the newly created environment

```shell
# Activates the previously created eos-project Anaconda environment.
conda activate eos-project
```

- Install some conda available packages on the environment

```shell
# We install matplotlib for plotting purposes
conda install matplotlib
```



### Configuring the environment from the YML file

The anaconda Ubuntu environment is also exported to the `eos-project-environment-ubuntu.yml` YML file. This file is available [here](environments/eos-project-environment-ubuntu.yml). You can load it in via the terminal as follows:


```shell
# Navigate to the folder where the YML file is located
cd eos/documentation/installation/environments
# Configure a new environment from the YML file
conda env create -f eos-project-environment-ubuntu.yml
# The export was done using
#   conda env export --no-build > eos-project-environment-ubuntu.yml
```

* * *
* * *
© [Lennert Bontinck](https://www.lennertbontinck.com/) VUB 2021-2022