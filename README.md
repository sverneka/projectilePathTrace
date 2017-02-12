# projectilePathTrace
Linear regression and LSTM implementation to trace trajectory of a projectile.

# Dependencies
  You need to install follwing packages on python 2.7
  - pandas, numpy, scipy, sklearn, keras(version 1.2.2)

#Run
python train.py

#Plot
To plot data, uncomment the commented out code in train.py meant for plotting.

#Results
predict path for projectile shot at an angle of 45 degrees and velocity of 10m/s
  The results will be saved in result_lin_reg.csv, result_lin_reg_log.csv, result_lstm.csv for respective models. Training insights are discussed in Report.pdf
