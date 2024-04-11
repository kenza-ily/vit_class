# Applied Deep Learning Coursework - COMP0197
The models are available in `.pth` in the linked OneDrive [folder](https://liveuclac-my.sharepoint.com/:f:/g/personal/ucabkbe_ucl_ac_uk/EulLst5l0HpOmMG06tsznX0BTCqFRl23EwQB37UC3fVRBA?e=GakjAC)

IMPORTANT: As the goals of the coursework is to "to develop the numerical algorithms in Python and one of the deep learning libraries TensorFlow and PyTorch.", I took the liberty of using the `torchmetrics` library, as it belongs to PyTorch, even if it wasn't proposed by the created library. This liberty was taken to prioritise learning objective.

This readme gives the outputed values obtained.

## Task 1 prints

### task.py

x_train: tensor([ -0.1497,  10.7289, -16.4609, -14.7188,  -7.7031,   5.3631,  -0.3963,
         15.8578,  -1.7749,   5.2923,  -6.0443,  -3.9313, -19.1070, -13.2456,
         -8.2445,   0.7409,   7.9067,  12.0005, -13.5588,  -8.7093])
y_train_gt: tensor([7.6779e-01, 3.6778e+02, 7.8096e+02, 6.2149e+02, 1.6361e+02, 9.8016e+01,
        6.7855e-01, 7.8712e+02, 6.9008e+00, 9.5608e+01, 9.8511e+01, 3.9503e+01,
        1.0580e+03, 5.0085e+02, 1.8842e+02, 4.1284e+00, 2.0436e+02, 4.5703e+02,
        5.2541e+02, 2.1113e+02])
t_train: tensor([1.0672e+00, 3.6701e+02, 7.8079e+02, 6.2242e+02, 1.6350e+02, 9.7645e+01,
        9.5991e-01, 7.8725e+02, 6.8139e+00, 9.5269e+01, 9.8980e+01, 3.9747e+01,
        1.0586e+03, 5.0089e+02, 1.8782e+02, 4.1260e+00, 2.0410e+02, 4.5688e+02,
        5.2462e+02, 2.1199e+02])
------
x_test: tensor([  3.7269, -15.5061, -13.8617, -10.3317,   9.0495,   8.0432, -11.8470,
          6.0421,  10.9794,  -2.5243])
y_test_gt: tensor([ 50.1227, 691.3062, 549.7187, 300.5670, 264.7772, 211.1661, 398.3636,
        122.6067, 384.6032,  15.0683])
t_test: tensor([ 50.4515, 691.7755, 550.2845, 300.2442, 263.8920, 211.2732, 398.0945,
        122.9007, 385.4062,  15.2822])
LEAST SQUARE EVALUATION
M=2
a) Difference between observed training data and true polynomial curve
Mean difference between observed training data and true polynomial curve: 0.009434586390852928
Standard deviation of difference between observed training data and true polynomial curve: 0.48098304867744446
b) Difference between LS-predicted values and true polynomial curve
Mean difference between LS-predicted values and true polynomial curve: -212.5789337158203
Standard deviation of difference between LS-predicted values and true polynomial curve: 217.66619873046875
---------------------------------------------------
M=3
a) Difference between observed training data and true polynomial curve
Mean difference between observed training data and true polynomial curve: 0.009434586390852928
Standard deviation of difference between observed training data and true polynomial curve: 0.48098304867744446
b) Difference between LS-predicted values and true polynomial curve
Mean difference between LS-predicted values and true polynomial curve: -751.3699951171875
Standard deviation of difference between LS-predicted values and true polynomial curve: 2249.113037109375
---------------------------------------------------
M=4
a) Difference between observed training data and true polynomial curve
Mean difference between observed training data and true polynomial curve: 0.009434586390852928
Standard deviation of difference between observed training data and true polynomial curve: 0.48098304867744446
b) Difference between LS-predicted values and true polynomial curve
Mean difference between LS-predicted values and true polynomial curve: 22818.490234375
Standard deviation of difference between LS-predicted values and true polynomial curve: 35477.3515625
---------------------------------------------------
Question
SGD EVALUATION
M=2
Epoch 0: Loss = 483663.90625
Epoch 100: Loss = 516.4835205078125
Epoch 200: Loss = 243.8001708984375
Epoch 300: Loss = 18.345611572265625
Epoch 400: Loss = 4.459098815917969
Epoch 500: Loss = 3.085139751434326
Epoch 600: Loss = 0.2722330391407013
Epoch 700: Loss = 0.2836449146270752
Epoch 800: Loss = 0.7893298864364624
Epoch 900: Loss = 0.8956307172775269
Difference between observed training data and true polynomial curve
Mean difference between SGD-predicted values and true polynomial curve for M=2: 64.06350708007812
Standard deviation of difference between SGD-predicted values and true polynomial curve for M=2: 65.0047607421875
Difference between the SGD-predicted values and the true polynomial curve for the test set
Mean difference between SGD-predicted values and true polynomial curve for M=2 (test set): 61.19587326049805
Standard deviation of difference between SGD-predicted values and true polynomial curve for M=2 (test set): 46.45047378540039
---------------------------------------------------
M=3
Epoch 0: Loss = 2275883.25
Epoch 100: Loss = 7521.3984375
Epoch 200: Loss = 404.2327575683594
Epoch 300: Loss = 114.84721374511719
Epoch 400: Loss = 4.028972148895264
Epoch 500: Loss = 88.20793914794922
Epoch 600: Loss = 13.20462417602539
Epoch 700: Loss = 904.3511352539062
Epoch 800: Loss = 206.86563110351562
Epoch 900: Loss = 26.208139419555664
Difference between observed training data and true polynomial curve
Mean difference between SGD-predicted values and true polynomial curve for M=3: -3764.796875
Standard deviation of difference between SGD-predicted values and true polynomial curve for M=3: 12969.755859375
Difference between the SGD-predicted values and the true polynomial curve for the test set
Mean difference between SGD-predicted values and true polynomial curve for M=3 (test set): -3479.52197265625
Standard deviation of difference between SGD-predicted values and true polynomial curve for M=3 (test set): 9027.669921875
---------------------------------------------------
M=4
Epoch 0: Loss = 6254837.5
Epoch 100: Loss = 404836.1875
Epoch 200: Loss = 52648.02734375
Epoch 300: Loss = 4505.0263671875
Epoch 400: Loss = 21590.880859375
Epoch 500: Loss = 1880.7874755859375
Epoch 600: Loss = 0.26611489057540894
Epoch 700: Loss = 5310.25927734375
Epoch 800: Loss = 16396934.0
Epoch 900: Loss = 138139.578125
Difference between observed training data and true polynomial curve
Mean difference between SGD-predicted values and true polynomial curve for M=4: 54620.2109375
Standard deviation of difference between SGD-predicted values and true polynomial curve for M=4: 85768.8046875
Difference between the SGD-predicted values and the true polynomial curve for the test set
Mean difference between SGD-predicted values and true polynomial curve for M=4 (test set): 38063.03515625
Standard deviation of difference between SGD-predicted values and true polynomial curve for M=4 (test set): 45801.02734375
---------------------------------------------------
SPEED COMPARISON
Least squares
Time spent in fitting using least squares: 0.014287948608398438 seconds
Time spent in training using least squares: 0.0006489753723144531 seconds
-----------------
SGD times
Time spent in fitting using SGD: 4.363103866577148 seconds
Time spent in training using SGD: 0.0002772808074951172 seconds

### task1a.py

x_train: tensor([ -0.1497,  10.7289, -16.4609, -14.7188,  -7.7031,   5.3631,  -0.3963,
         15.8578,  -1.7749,   5.2923,  -6.0443,  -3.9313, -19.1070, -13.2456,
         -8.2445,   0.7409,   7.9067,  12.0005, -13.5588,  -8.7093])
y_train_gt: tensor([7.6779e-01, 3.6778e+02, 7.8096e+02, 6.2149e+02, 1.6361e+02, 9.8016e+01,
        6.7855e-01, 7.8712e+02, 6.9008e+00, 9.5608e+01, 9.8511e+01, 3.9503e+01,
        1.0580e+03, 5.0085e+02, 1.8842e+02, 4.1284e+00, 2.0436e+02, 4.5703e+02,
        5.2541e+02, 2.1113e+02])
t_train: tensor([1.0672e+00, 3.6701e+02, 7.8079e+02, 6.2242e+02, 1.6350e+02, 9.7645e+01,
        9.5991e-01, 7.8725e+02, 6.8139e+00, 9.5269e+01, 9.8980e+01, 3.9747e+01,
        1.0586e+03, 5.0089e+02, 1.8782e+02, 4.1260e+00, 2.0410e+02, 4.5688e+02,
        5.2462e+02, 2.1199e+02])
------
x_test: tensor([  3.7269, -15.5061, -13.8617, -10.3317,   9.0495,   8.0432, -11.8470,
          6.0421,  10.9794,  -2.5243])
y_test_gt: tensor([ 50.1227, 691.3062, 549.7187, 300.5670, 264.7772, 211.1661, 398.3636,
        122.6067, 384.6032,  15.0683])
t_test: tensor([ 50.4515, 691.7755, 550.2845, 300.2442, 263.8920, 211.2732, 398.0945,
        122.9007, 385.4062,  15.2822])
Epoch 0: Loss = 483663.90625
Epoch 100: Loss = 516.4835205078125
Epoch 200: Loss = 243.8001708984375
Epoch 300: Loss = 18.345611572265625
Epoch 400: Loss = 4.459098815917969
Epoch 500: Loss = 3.085139751434326
Epoch 600: Loss = 0.2722330391407013
Epoch 700: Loss = 0.2836449146270752
Epoch 800: Loss = 0.7893298864364624
Epoch 900: Loss = 0.8956307172775269
Optimized value of M: 2
Difference between observed training data and true polynomial curve
Mean difference between SGD-predicted values and true polynomial curve: 64.06350708007812
Standard deviation of difference between SGD-predicted values and true polynomial curve: 65.0047607421875
Difference between the SGD-predicted values and the true polynomial curve for the test set
Mean difference between SGD-predicted values and true polynomial curve (test set): 61.19587326049805
Standard deviation of difference between SGD-predicted values and true polynomial curve (test set): 46.45047378540039




## Task 2 prints

### task.py

<!-- The prints were filtered here as the original output was a line per batch, so more than 45000lines -->

Sampling_method=1
Epoch: 1, Training Loss: 0.678
Epoch: 2, Training Loss: 0.534
Epoch: 3, Training Loss: 0.484
Epoch: 4, Training Loss: 0.452
Epoch: 5, Training Loss: 0.437
Epoch: 6, Training Loss: 0.420
Epoch: 7, Training Loss: 0.407
Epoch: 8, Training Loss: 0.398
Epoch: 9, Training Loss: 0.395
Epoch: 10, Training Loss: 0.388
Epoch: 11, Training Loss: 0.386
Epoch: 12, Training Loss: 0.380
Epoch: 13, Training Loss: 0.378
Epoch: 14, Training Loss: 0.373
Epoch: 15, Training Loss: 0.376
Epoch: 16, Training Loss: 0.372
Epoch: 17, Training Loss: 0.370
Epoch: 18, Training Loss: 0.368
Epoch: 19, Training Loss: 0.365
Epoch: 20, Training Loss: 0.361
Saved trained model

Accuracy of the model on the test images: 91.2%
Image saved at result_1.png


Sampling_method=2
Epoch: 1, Training Loss: 0.925
Epoch: 2, Training Loss: 0.874
Epoch: 3, Training Loss: 0.847
Epoch: 4, Training Loss: 0.821
Epoch: 5, Training Loss: 0.795
Epoch: 6, Training Loss: 0.768
Epoch: 7, Training Loss: 0.742
Epoch: 8, Training Loss: 0.716
Epoch: 9, Training Loss: 0.689
Epoch: 10, Training Loss: 0.663
Epoch: 11, Training Loss: 0.637
Epoch: 12, Training Loss: 0.611
Epoch: 13, Training Loss: 0.584
Epoch: 14, Training Loss: 0.558
Epoch: 15, Training Loss: 0.532
Epoch: 16, Training Loss: 0.505
Epoch: 17, Training Loss: 0.479
Epoch: 18, Training Loss: 0.453
Epoch: 19, Training Loss: 0.426
Epoch: 20, Training Loss: 0.396
Saved trained model

Accuracy of the model on the test images: 86.3%
Image saved at result_2.png


## Task 3 prints

### task.py

Sampling_method=1
Files have been successfully downloaded and verified.
Total batches in train_loader: 1125
Total batches in val_loader: 125
Total batches in test_loader: 1125
Epoch: 1, Batch: 100, Batch Loss: 1.082
Epoch: 1, Batch: 200, Batch Loss: 1.270
Epoch: 1, Batch: 300, Batch Loss: 1.165
Epoch: 1, Batch: 400, Batch Loss: 1.072
Epoch: 1, Batch: 500, Batch Loss: 1.054
Epoch: 1, Batch: 600, Batch Loss: 0.932
Epoch: 1, Batch: 700, Batch Loss: 1.095
Epoch: 1, Batch: 800, Batch Loss: 1.050
Epoch: 1, Batch: 900, Batch Loss: 1.062
Epoch: 1, Batch: 1000, Batch Loss: 0.960
Epoch: 1, Batch: 1100, Batch Loss: 0.908
Epoch: 1, Cumulative Training Loss: 1.085
Validation Loss: 0.238, Validation AUC-ROC: 0.999, Validation Accuracy: 0.956
Epoch: 2, Batch: 100, Batch Loss: 0.880
Epoch: 2, Batch: 200, Batch Loss: 0.863
Epoch: 2, Batch: 300, Batch Loss: 1.105
Epoch: 2, Batch: 400, Batch Loss: 0.972
Epoch: 2, Batch: 500, Batch Loss: 0.940
Epoch: 2, Batch: 600, Batch Loss: 0.810
Epoch: 2, Batch: 700, Batch Loss: 0.698
Epoch: 2, Batch: 800, Batch Loss: 1.030
Epoch: 2, Batch: 900, Batch Loss: 1.083
Epoch: 2, Batch: 1000, Batch Loss: 0.946
Epoch: 2, Batch: 1100, Batch Loss: 0.963
Epoch: 2, Cumulative Training Loss: 0.955
Validation Loss: 0.182, Validation AUC-ROC: 0.999, Validation Accuracy: 0.964
Epoch: 3, Batch: 100, Batch Loss: 1.098
Epoch: 3, Batch: 200, Batch Loss: 0.880
Epoch: 3, Batch: 300, Batch Loss: 0.855
Epoch: 3, Batch: 400, Batch Loss: 0.900
Epoch: 3, Batch: 500, Batch Loss: 0.829
Epoch: 3, Batch: 600, Batch Loss: 0.965
Epoch: 3, Batch: 700, Batch Loss: 0.840
Epoch: 3, Batch: 800, Batch Loss: 0.896
Epoch: 3, Batch: 900, Batch Loss: 0.985
Epoch: 3, Batch: 1000, Batch Loss: 0.944
Epoch: 3, Batch: 1100, Batch Loss: 0.870
Epoch: 3, Cumulative Training Loss: 0.910
Validation Loss: 0.174, Validation AUC-ROC: 0.999, Validation Accuracy: 0.968
Epoch: 4, Batch: 100, Batch Loss: 0.843
Epoch: 4, Batch: 200, Batch Loss: 1.028
Epoch: 4, Batch: 300, Batch Loss: 0.593
Epoch: 4, Batch: 400, Batch Loss: 0.774
Epoch: 4, Batch: 500, Batch Loss: 0.996
Epoch: 4, Batch: 600, Batch Loss: 0.759
Epoch: 4, Batch: 700, Batch Loss: 0.677
Epoch: 4, Batch: 800, Batch Loss: 1.026
Epoch: 4, Batch: 900, Batch Loss: 0.870
Epoch: 4, Batch: 1000, Batch Loss: 0.732
Epoch: 4, Batch: 1100, Batch Loss: 0.853
Epoch: 4, Cumulative Training Loss: 0.877
Validation Loss: 0.166, Validation AUC-ROC: 0.999, Validation Accuracy: 0.967
Epoch: 5, Batch: 100, Batch Loss: 0.765
Epoch: 5, Batch: 200, Batch Loss: 0.783
Epoch: 5, Batch: 300, Batch Loss: 0.912
Epoch: 5, Batch: 400, Batch Loss: 0.904
Epoch: 5, Batch: 500, Batch Loss: 0.789
Epoch: 5, Batch: 600, Batch Loss: 0.948
Epoch: 5, Batch: 700, Batch Loss: 0.793
Epoch: 5, Batch: 800, Batch Loss: 0.840
Epoch: 5, Batch: 900, Batch Loss: 0.874
Epoch: 5, Batch: 1000, Batch Loss: 0.804
Epoch: 5, Batch: 1100, Batch Loss: 0.654
Epoch: 5, Cumulative Training Loss: 0.853
Validation Loss: 0.142, Validation AUC-ROC: 0.999, Validation Accuracy: 0.972
Epoch: 6, Batch: 100, Batch Loss: 0.779
Epoch: 6, Batch: 200, Batch Loss: 0.723
Epoch: 6, Batch: 300, Batch Loss: 0.943
Epoch: 6, Batch: 400, Batch Loss: 0.743
Epoch: 6, Batch: 500, Batch Loss: 0.806
Epoch: 6, Batch: 600, Batch Loss: 1.022
Epoch: 6, Batch: 700, Batch Loss: 0.764
Epoch: 6, Batch: 800, Batch Loss: 0.730
Epoch: 6, Batch: 900, Batch Loss: 0.902
Epoch: 6, Batch: 1000, Batch Loss: 0.767
Epoch: 6, Batch: 1100, Batch Loss: 1.003
Epoch: 6, Cumulative Training Loss: 0.836
Validation Loss: 0.164, Validation AUC-ROC: 0.999, Validation Accuracy: 0.967
...
Epoch: 20, Batch: 100, Batch Loss: 0.664
Epoch: 20, Batch: 200, Batch Loss: 0.798
Epoch: 20, Batch: 300, Batch Loss: 0.809
Epoch: 20, Batch: 400, Batch Loss: 0.910
Epoch: 20, Batch: 500, Batch Loss: 0.680
Epoch: 20, Batch: 600, Batch Loss: 0.771
Epoch: 20, Batch: 700, Batch Loss: 0.768
Epoch: 20, Batch: 800, Batch Loss: 0.753
Epoch: 20, Batch: 900, Batch Loss: 0.570
Epoch: 20, Batch: 1000, Batch Loss: 0.804
Epoch: 20, Batch: 1100, Batch Loss: 0.873
Epoch: 20, Cumulative Training Loss: 0.748
Validation Loss: 0.140, Validation AUC-ROC: 0.999, Validation Accuracy: 0.975
Validation AUC-ROC: 0.999
Validation Accuracy: 0.977

Using Sampling Method: 1
Test Loss: 0.062
Test AUC-ROC: 1.000
Test Accuracy: 1.000

Comparing with Development Results:
Lowest Validation Loss: 0.140
Highest Validation AUC-ROC: 0.999
Highest Validation Accuracy: 0.975

Sampling_method=2
Files already downloaded and verified
Number of batches in train_loader: 1125
Number of batches in val_loader: 125
Number of batches in test_loader: 1125
Epoch: 1, Batch: 100, Batch Loss: 1.095
Epoch: 1, Batch: 200, Batch Loss: 1.280
Epoch: 1, Batch: 300, Batch Loss: 1.178
Epoch: 1, Batch: 400, Batch Loss: 1.084
Epoch: 1, Batch: 500, Batch Loss: 1.067
Epoch: 1, Batch: 600, Batch Loss: 0.945
Epoch: 1, Batch: 700, Batch Loss: 1.107
Epoch: 1, Batch: 800, Batch Loss: 1.061
Epoch: 1, Batch: 900, Batch Loss: 1.073
Epoch: 1, Batch: 1000, Batch Loss: 0.970
Epoch: 1, Batch: 1100, Batch Loss: 0.918
Epoch: 1, Training Loss: 1.095
Validation Loss: 0.242, Validation AUC-ROC: 0.998, Validation Accuracy: 0.953
Epoch: 2, Batch: 100, Batch Loss: 0.892
Epoch: 2, Batch: 200, Batch Loss: 0.875
Epoch: 2, Batch: 300, Batch Loss: 1.118
Epoch: 2, Batch: 400, Batch Loss: 0.983
Epoch: 2, Batch: 500, Batch Loss: 0.952
Epoch: 2, Batch: 600, Batch Loss: 0.820
Epoch: 2, Batch: 700, Batch Loss: 0.708
Epoch: 2, Batch: 800, Batch Loss: 1.041
Epoch: 2, Batch: 900, Batch Loss: 1.093
Epoch: 2, Batch: 1000, Batch Loss: 0.956
Epoch: 2, Batch: 1100, Batch Loss: 0.973
Epoch: 2, Training Loss: 0.962
Validation Loss: 0.186, Validation AUC-ROC: 0.999, Validation Accuracy: 0.962
Epoch: 3, Batch: 100, Batch Loss: 1.111
Epoch: 3, Batch: 200, Batch Loss: 0.892
Epoch: 3, Batch: 300, Batch Loss: 0.867
Epoch: 3, Batch: 400, Batch Loss: 0.912
Epoch: 3, Batch: 500, Batch Loss: 0.841
Epoch: 3, Batch: 600, Batch Loss: 0.975
Epoch: 3, Batch: 700, Batch Loss: 0.850
Epoch: 3, Batch: 800, Batch Loss: 0.906
Epoch: 3, Batch: 900, Batch Loss: 0.995
Epoch: 3, Batch: 1000, Batch Loss: 0.954
Epoch: 3, Batch: 1100, Batch Loss: 0.880
Epoch: 3, Training Loss: 0.917
Validation Loss: 0.177, Validation AUC-ROC: 0.998, Validation Accuracy: 0.966
Epoch: 4, Batch: 100, Batch Loss: 0.853
Epoch: 4, Batch: 200, Batch Loss: 1.039
Epoch: 4, Batch: 300, Batch Loss: 0.603
Epoch: 4, Batch: 400, Batch Loss: 0.784
Epoch: 4, Batch: 500, Batch Loss: 1.006
Epoch: 4, Batch: 600, Batch Loss: 0.769
Epoch: 4, Batch: 700, Batch Loss: 0.687
Epoch: 4, Batch: 800, Batch Loss: 1.036
Epoch: 4, Batch: 900, Batch Loss: 0.880
Epoch: 4, Batch: 1000, Batch Loss: 0.742
Epoch: 4, Batch: 1100, Batch Loss: 0.863
Epoch: 4, Training Loss: 0.884
Validation Loss: 0.170, Validation AUC-ROC: 0.998, Validation Accuracy: 0.965
Epoch: 5, Batch: 100, Batch Loss: 0.775
Epoch: 5, Batch: 200, Batch Loss: 0.793
Epoch: 5, Batch: 300, Batch Loss: 0.922
Epoch: 5, Batch: 400, Batch Loss: 0.914
Epoch: 5, Batch: 500, Batch Loss: 0.799
Epoch: 5, Batch: 600, Batch Loss: 0.958
Epoch: 5, Batch: 700, Batch Loss: 0.803
Epoch: 5, Batch: 800, Batch Loss: 0.850
Epoch: 5, Batch: 900, Batch Loss: 0.884
Epoch: 5, Batch: 1000, Batch Loss: 0.814
Epoch: 5, Batch: 1100, Batch Loss: 0.664
Epoch: 5, Training Loss: 0.860
Validation Loss: 0.146, Validation AUC-ROC: 0.999, Validation Accuracy: 0.970
Epoch: 6, Batch: 100, Batch Loss: 0.789
Epoch: 6, Batch: 200, Batch Loss: 0.733
Epoch: 6, Batch: 300, Batch Loss: 0.953
Epoch: 6, Batch: 400, Batch Loss: 0.753
Epoch: 6, Batch: 500, Batch Loss: 0.816
Epoch: 6, Batch: 600, Batch Loss: 1.034
Epoch: 6, Batch: 700, Batch Loss: 0.774
Epoch: 6, Batch: 800, Batch Loss: 0.740
Epoch: 6, Batch: 900, Batch Loss: 0.912
Epoch: 6, Batch: 1000, Batch Loss: 0.777
Epoch: 6, Batch: 1100, Batch Loss: 1.013
Epoch: 6, Training Loss: 0.843
Validation Loss: 0.168, Validation AUC-ROC: 0.998, Validation Accuracy: 0.965
Epoch: 7, Batch: 100, Batch Loss: 0.817
Epoch: 7, Batch: 200, Batch Loss: 0.785
Epoch: 7, Batch: 300, Batch Loss: 0.875
Epoch: 7, Batch: 400, Batch Loss: 0.706
Epoch: 7, Batch: 500, Batch Loss: 0.704
Epoch: 7, Batch: 600, Batch Loss: 0.820
Epoch: 7, Batch: 700, Batch Loss: 0.963
Epoch: 7, Batch: 800, Batch Loss: 0.816
Epoch: 7, Batch: 900, Batch Loss: 0.730
Epoch: 7, Batch: 1000, Batch Loss: 1.025
Epoch: 7, Batch: 1100, Batch Loss: 0.813
Epoch: 7, Training Loss: 0.830
Validation Loss: 0.145, Validation AUC-ROC: 0.999, Validation Accuracy: 0.972
Epoch: 8, Batch: 100, Batch Loss: 0.826
Epoch: 8, Batch: 200, Batch Loss: 0.714
Epoch: 8, Batch: 300, Batch Loss: 0.883
Epoch: 8, Batch: 400, Batch Loss: 0.727
Epoch: 8, Batch: 500, Batch Loss: 0.965
Epoch: 8, Batch: 600, Batch Loss: 0.844
Epoch: 8, Batch: 700, Batch Loss: 0.794
Epoch: 8, Batch: 800, Batch Loss: 0.874
Epoch: 8, Batch: 900, Batch Loss: 0.836
Epoch: 8, Batch: 1000, Batch Loss: 0.936
Epoch: 8, Batch: 1100, Batch Loss: 0.682
Epoch: 8, Training Loss: 0.825
Validation Loss: 0.163, Validation AUC-ROC: 0.998, Validation Accuracy: 0.969
Epoch: 9, Batch: 100, Batch Loss: 0.704
Epoch: 9, Batch: 200, Batch Loss: 0.716
Epoch: 9, Batch: 300, Batch Loss: 0.749
Epoch: 9, Batch: 400, Batch Loss: 0.715
Epoch: 9, Batch: 500, Batch Loss: 0.950
Epoch: 9, Batch: 600, Batch Loss: 0.707
Epoch: 9, Batch: 700, Batch Loss: 0.782
Epoch: 9, Batch: 800, Batch Loss: 0.889
Epoch: 9, Batch: 900, Batch Loss: 0.574
Epoch: 9, Batch: 1000, Batch Loss: 0.792
Epoch: 9, Batch: 1100, Batch Loss: 0.640
Epoch: 9, Training Loss: 0.815
Validation Loss: 0.161, Validation AUC-ROC: 0.998, Validation Accuracy: 0.970
Epoch: 10, Batch: 100, Batch Loss: 0.813
Epoch: 10, Batch: 200, Batch Loss: 0.834
Epoch: 10, Batch: 300, Batch Loss: 0.688
Epoch: 10, Batch: 400, Batch Loss: 0.848
Epoch: 10, Batch: 500, Batch Loss: 0.643
Epoch: 10, Batch: 600, Batch Loss: 0.668
Epoch: 10, Batch: 700, Batch Loss: 1.141
Epoch: 10, Batch: 800, Batch Loss: 0.777
Epoch: 10, Batch: 900, Batch Loss: 0.755
Epoch: 10, Batch: 1000, Batch Loss: 0.892
Epoch: 10, Batch: 1100, Batch Loss: 0.948
Epoch: 10, Training Loss: 0.804
Validation Loss: 0.147, Validation AUC-ROC: 0.998, Validation Accuracy: 0.971
Epoch: 11, Batch: 100, Batch Loss: 0.793
Epoch: 11, Batch: 200, Batch Loss: 0.664
Epoch: 11, Batch: 300, Batch Loss: 0.727
Epoch: 11, Batch: 400, Batch Loss: 0.729
Epoch: 11, Batch: 500, Batch Loss: 0.711
Epoch: 11, Batch: 600, Batch Loss: 0.731
Epoch: 11, Batch: 700, Batch Loss: 0.939
Epoch: 11, Batch: 800, Batch Loss: 0.747
Epoch: 11, Batch: 900, Batch Loss: 0.666
Epoch: 11, Batch: 1000, Batch Loss: 0.794
Epoch: 11, Batch: 1100, Batch Loss: 0.651
Epoch: 11, Training Loss: 0.794
Validation Loss: 0.147, Validation AUC-ROC: 0.998, Validation Accuracy: 0.974
Epoch: 12, Batch: 100, Batch Loss: 0.821
Epoch: 12, Batch: 200, Batch Loss: 0.715
Epoch: 12, Batch: 300, Batch Loss: 0.907
Epoch: 12, Batch: 400, Batch Loss: 0.761
Epoch: 12, Batch: 500, Batch Loss: 0.902
Epoch: 12, Batch: 600, Batch Loss: 0.639
Epoch: 12, Batch: 700, Batch Loss: 0.792
Epoch: 12, Batch: 800, Batch Loss: 1.008
Epoch: 12, Batch: 900, Batch Loss: 0.760
Epoch: 12, Batch: 1000, Batch Loss: 0.824
Epoch: 12, Batch: 1100, Batch Loss: 0.749
Epoch: 12, Training Loss: 0.792
Validation Loss: 0.143, Validation AUC-ROC: 0.998, Validation Accuracy: 0.975
Epoch: 13, Batch: 100, Batch Loss: 0.791
Epoch: 13, Batch: 200, Batch Loss: 0.645
Epoch: 13, Batch: 300, Batch Loss: 0.677
Epoch: 13, Batch: 400, Batch Loss: 0.864
Epoch: 13, Batch: 500, Batch Loss: 0.712
Epoch: 13, Batch: 600, Batch Loss: 0.834
Epoch: 13, Batch: 700, Batch Loss: 0.815
Epoch: 13, Batch: 800, Batch Loss: 0.748
Epoch: 13, Batch: 900, Batch Loss: 0.712
Epoch: 13, Batch: 1000, Batch Loss: 0.692
Epoch: 13, Batch: 1100, Batch Loss: 0.778
Epoch: 13, Training Loss: 0.785
Validation Loss: 0.140, Validation AUC-ROC: 0.998, Validation Accuracy: 0.974
Epoch: 14, Batch: 100, Batch Loss: 0.723
Epoch: 14, Batch: 200, Batch Loss: 0.902
Epoch: 14, Batch: 300, Batch Loss: 0.843
Epoch: 14, Batch: 400, Batch Loss: 0.784
Epoch: 14, Batch: 500, Batch Loss: 0.687
Epoch: 14, Batch: 600, Batch Loss: 0.787
Epoch: 14, Batch: 700, Batch Loss: 0.584
Epoch: 14, Batch: 800, Batch Loss: 0.930
Epoch: 14, Batch: 900, Batch Loss: 0.744
Epoch: 14, Batch: 1000, Batch Loss: 0.998
Epoch: 14, Batch: 1100, Batch Loss: 0.742
Epoch: 14, Training Loss: 0.780
Validation Loss: 0.136, Validation AUC-ROC: 0.998, Validation Accuracy: 0.975
Epoch: 15, Batch: 100, Batch Loss: 0.890
Epoch: 15, Batch: 200, Batch Loss: 0.692
Epoch: 15, Batch: 300, Batch Loss: 0.800
Epoch: 15, Batch: 400, Batch Loss: 0.716
Epoch: 15, Batch: 500, Batch Loss: 0.843
Epoch: 15, Batch: 600, Batch Loss: 0.818
Epoch: 15, Batch: 700, Batch Loss: 0.922
Epoch: 15, Batch: 800, Batch Loss: 0.525
Epoch: 15, Batch: 900, Batch Loss: 0.830
Epoch: 15, Batch: 1000, Batch Loss: 0.791
Epoch: 15, Batch: 1100, Batch Loss: 0.881
Epoch: 15, Training Loss: 0.773
Validation Loss: 0.146, Validation AUC-ROC: 0.997, Validation Accuracy: 0.971
Epoch: 16, Batch: 100, Batch Loss: 0.732
Epoch: 16, Batch: 200, Batch Loss: 0.663
Epoch: 16, Batch: 300, Batch Loss: 0.930
Epoch: 16, Batch: 400, Batch Loss: 0.954
Epoch: 16, Batch: 500, Batch Loss: 0.756
Epoch: 16, Batch: 600, Batch Loss: 0.755
Epoch: 16, Batch: 700, Batch Loss: 0.886
Epoch: 16, Batch: 800, Batch Loss: 0.781
Epoch: 16, Batch: 900, Batch Loss: 0.751
Epoch: 16, Batch: 1000, Batch Loss: 0.781
Epoch: 16, Batch: 1100, Batch Loss: 0.773
Epoch: 16, Training Loss: 0.768
Validation Loss: 0.147, Validation AUC-ROC: 0.998, Validation Accuracy: 0.970
Epoch: 17, Batch: 100, Batch Loss: 0.889
Epoch: 17, Batch: 200, Batch Loss: 0.669
Epoch: 17, Batch: 300, Batch Loss: 0.803
Epoch: 17, Batch: 400, Batch Loss: 0.811
Epoch: 17, Batch: 500, Batch Loss: 0.782
Epoch: 17, Batch: 600, Batch Loss: 0.639
Epoch: 17, Batch: 700, Batch Loss: 0.780
Epoch: 17, Batch: 800, Batch Loss: 0.668
Epoch: 17, Batch: 900, Batch Loss: 0.894
Epoch: 17, Batch: 1000, Batch Loss: 0.901
Epoch: 17, Batch: 1100, Batch Loss: 0.726
Epoch: 17, Training Loss: 0.764
Validation Loss: 0.138, Validation AUC-ROC: 0.999, Validation Accuracy: 0.973
Epoch: 18, Batch: 100, Batch Loss: 0.925
Epoch: 18, Batch: 200, Batch Loss: 0.680
Epoch: 18, Batch: 300, Batch Loss: 0.644
Epoch: 18, Batch: 400, Batch Loss: 0.740
Epoch: 18, Batch: 500, Batch Loss: 0.776
Epoch: 18, Batch: 600, Batch Loss: 0.789
Epoch: 18, Batch: 700, Batch Loss: 0.701
Epoch: 18, Batch: 800, Batch Loss: 0.843
Epoch: 18, Batch: 900, Batch Loss: 0.896
Epoch: 18, Batch: 1000, Batch Loss: 0.824
Epoch: 18, Batch: 1100, Batch Loss: 0.704
Epoch: 18, Training Loss: 0.766
Validation Loss: 0.137, Validation AUC-ROC: 0.998, Validation Accuracy: 0.975
Epoch: 19, Batch: 100, Batch Loss: 0.615
Epoch: 19, Batch: 200, Batch Loss: 0.593
Epoch: 19, Batch: 300, Batch Loss: 0.729
Epoch: 19, Batch: 400, Batch Loss: 0.771
Epoch: 19, Batch: 500, Batch Loss: 0.764
Epoch: 19, Batch: 600, Batch Loss: 0.687
Epoch: 19, Batch: 700, Batch Loss: 0.823
Epoch: 19, Batch: 800, Batch Loss: 0.839
Epoch: 19, Batch: 900, Batch Loss: 0.797
Epoch: 19, Batch: 1000, Batch Loss: 0.702
Epoch: 19, Batch: 1100, Batch Loss: 0.800
Epoch: 19, Training Loss: 0.758
Validation Loss: 0.125, Validation AUC-ROC: 0.999, Validation Accuracy: 0.975
Epoch: 20, Batch: 100, Batch Loss: 0.674
Epoch: 20, Batch: 200, Batch Loss: 0.808
Epoch: 20, Batch: 300, Batch Loss: 0.819
Epoch: 20, Batch: 400, Batch Loss: 0.920
Epoch: 20, Batch: 500, Batch Loss: 0.690
Epoch: 20, Batch: 600, Batch Loss: 0.781
Epoch: 20, Batch: 700, Batch Loss: 0.778
Epoch: 20, Batch: 800, Batch Loss: 0.763
Epoch: 20, Batch: 900, Batch Loss: 0.580
Epoch: 20, Batch: 1000, Batch Loss: 0.814
Epoch: 20, Batch: 1100, Batch Loss: 0.883
Epoch: 20, Training Loss: 0.756
Validation Loss: 0.144, Validation AUC-ROC: 0.998, Validation Accuracy: 0.973
Validation AUC-ROC: 0.999
Validation Accuracy: 0.975

Sampling Method: 2
Test Loss: 0.064
Test AUC-ROC: 1.000
Test Accuracy: 1.000

Comparison with Development Results:
Best Validation Loss: 0.125
Best Validation AUC-ROC: 0.999
Best Validation Accuracy: 0.975