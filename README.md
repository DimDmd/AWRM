# AWRM

Adaptive window rolling median, namely AWRM is a simple but dynamic algorithm for anomaly detection in time series. This algorithms spots abnormal data points using a rolling median with an adaptive sliding window. The window changes based on two methods, F1 based and T-test. F1 method tries to make the F1 score have only an upward trend, while T-test recognizes trends in time series and adjusts the window accordingly.

This is an implementation of the paper "Utilizing an adaptive window rolling median methodology for time series anomaly detection".
