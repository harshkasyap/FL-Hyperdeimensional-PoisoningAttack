# FL-Hyperdeimensional-PoisoningAttack

* If Conda enviroment
    * conda env create --name env_name --file=cenv.yml
        * it will also add this conda env in your base jupyter notebook, look for [reference](https://towardsdatascience.com/how-to-set-up-anaconda-and-jupyter-notebook-the-right-way-de3b7623ea4a)
    * conda activate env_name
    * open host_ip:port for opening jupyter notebook
        * jupyter notebook --no-browser --ip="*" --port=xxxx --NotebookApp.token='xx' --NotebookApp.iopub_data_rate_limit=1.0e1000
