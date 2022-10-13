from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
labels = [1, 0, 0, 1, 1, 1, 0, 1, 1, 1]	# 실제 labels
guesses = [0, 1, 1, 1, 1, 0, 1, 0, 1, 0]	# 에측된 결과

# 정확도 ->  올바르게 예측된 데이터의 수를 전체 데이터의 수로 나눈 값
# (TP+TN)/(TP+TN+FP+FN)
print(accuracy_score(labels, guesses))	# 0.3
# 재현율 -> 실제로 True인 데이터를 모델이 True라고 인식한 데이터의 수
# TP/(TP+FN)
print(recall_score(labels, guesses))	# 0.42
# 정밀도 -> 모델이 True로 예측한 데이터 중 실제로 True인 데이터의 수
# TP/(TP+FP)
print(precision_score(labels, guesses))	# 0.5
# 정밀도와 재현율의 조화평균
# 2*(Precision*Recall)/(Precision+Recall)
print(f1_score(labels, guesses))	# 0.46

# 참고 자료 
# https://eunsukimme.github.io/ml/2019/10/21/Accuracy-Recall-Precision-F1-score/
