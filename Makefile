help:
	@echo "make regression : For running Gradient Descent Regression model on random dataset"
	@echo "make polynomial_features : For testing tranformation built similar to sklearn’s polynomial preprocessing"
	@echo "make normal_regression : To check how theta vary with degree"
	@echo "make poly_theta : To check how theta vary with degree in polynomial features"
	@echo "make contour : For generating a contour of the gradient descent"
	@echo "make compare_time : For comparing time taken by normal regression and gradient descent"
	@echo "make collinear : For checking the feature dependency (collinear features)"

regression:
	@ python linear_regression_test.py 

polynomial_features:
	@ python poly_features_test.py

normal_regression:
	@ python Normal_regression.py 

poly_theta:
	@ python degreevstheta.py 

contour:
	@ python plot_contour.py 

compare_time:
	@ python compare_time.py 

collinear:
	@ python collinear_dataset.py 