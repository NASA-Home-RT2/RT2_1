STEP BY STEP PROCEDURE TO RUN GUI
by Heraldo Rozas (heraldo.rozas1@gmail.com)

0.	INSTALL GUROBI WITH ACADEMIC LICENSE
    •	https://support.gurobi.com/hc/en-us/articles/4534161999889-How-do-I-install-Gurobi-Optimizer
    •	https://www.gurobi.com/features/academic-named-user-license/
1.	INSTALL REQUIRED PYTHON PACKAGES
       conda install --file requirements.txt
2.	FIND DIRECTORY
   Go to the directory “joint_optimization_gui_demoA3_HOME”
3.	RUNNING THE CODES
  1.	Run the local server to see the GUI:
      a.	python server.py
  2.	Run the simulation model:
      a.	python simulation_optimization_engine.py
  3.	Wait for 5 minutes to let the simulation run.
  4.	See the GUI by visiting  http://localhost:3000/  on your Chrome browser in incognito mode.








