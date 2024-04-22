# Useful Commands

## Conda
To see a list of all your environments:
`conda info --envs`

To change your current environment:
`conda activate <env-name>`

To change your current environment back to the default base:
`conda activate`

The most basic way to create a new environment is with the following command:
`conda create -n <env-name>`

You can also install packages into a previously created environment. To do this, you can either activate the environment you want to modify or specify the environment name on the command line:
**via environment activation**
`conda activate myenvironment`
`conda install matplotlib`
**via command line option**
`conda install --name myenvironment matplotlib`

https://conda.io/projects/conda/en/latest/user-guide/getting-started.html
