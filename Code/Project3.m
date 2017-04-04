clc ;
clear ;

% 导入训练数据
Images_tr = loadMNISTImages('../Data/train-images.idx3-ubyte') ;
Labels_tr = loadMNISTLabels('../Data/train-labels.idx1-ubyte') ;

% 导入测试数据
Images_te = loadMNISTImages('../Data/t10k-images.idx3-ubyte') ;
Labels_te = loadMNISTLabels('../Data/t10k-labels.idx1-ubyte') ;

disp('数据导入完成')


% libsvm
% 训练样本模型
svm_model = libsvmtrain(Labels_tr, Images_tr') ;       % training
disp('模型训练完成')

% 使用模型预测测试数据
[predicted_label, accuracy, prob_estimates] = libsvmpredict(Labels_te, Images_te', svm_model) ;       % test
disp('数据预测完成')

% 显示正确率
disp(accuracy) ;