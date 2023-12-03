import pandas as pd
from scipy import stats

def read_and_analyze_statistics(model_file, model_type):
    # Read the CSV File
    data = pd.read_csv(model_file)

    # Calculate Mean and Standard Deviation for Accuracy and Log Loss
    accuracy_mean = data['Train Accuracy'].mean()
    accuracy_std = data['Train Accuracy'].std()
    log_loss_mean = data['Train Loss'].mean()
    log_loss_std = data['Train Loss'].std()

    print(f"{model_type} Accuracy: Mean = {accuracy_mean}, Standard Deviation = {accuracy_std}")
    print(f"{model_type} Log Loss: Mean = {log_loss_mean}, Standard Deviation = {log_loss_std}")

    return accuracy_mean, accuracy_std, log_loss_mean, log_loss_std

def compare_models(ann_file, rnn_file):
    ann_accuracy_mean, ann_accuracy_std, ann_log_loss_mean, ann_log_loss_std = read_and_analyze_statistics(ann_file, "ANN")
    rnn_accuracy_mean, rnn_accuracy_std, rnn_log_loss_mean, rnn_log_loss_std = read_and_analyze_statistics(rnn_file, "RNN")

    # Perform T-Test for Accuracy and Log Loss
    accuracy_t_test_result = stats.ttest_ind_from_stats(ann_accuracy_mean, ann_accuracy_std, 40, rnn_accuracy_mean, rnn_accuracy_std, 60)
    log_loss_t_test_result = stats.ttest_ind_from_stats(ann_log_loss_mean, ann_log_loss_std, 40, rnn_log_loss_mean, rnn_log_loss_std, 60)

    print(f"T-Test Result for Accuracy: {accuracy_t_test_result}")
    print(f"T-Test Result for Log Loss: {log_loss_t_test_result}")

if __name__ == '__main__':
    company = input("Please enter the name of the company to predict: Chevron or Exxon: ")
    if company != "Chevron" and company != "Exxon" and company != "both":
        print("Invalid company name")
        exit(1)

    if company == "Exxon":
        compare_models('exxon_ann_history_seed_194.csv', 'exxon_rnn_history_seed_222.csv')
    elif company == "Chevron":
        compare_models('chevron_ann_history_seed_87.csv', 'chevron_rnn_history_seed_4.csv')
    elif company == "both":
        print("Exxon:")
        compare_models('exxon_ann_history_seed_194.csv', 'exxon_rnn_history_seed_222.csv')
        print("Chevron:")
        compare_models('chevron_ann_history_seed_87.csv', 'chevron_rnn_history_seed_4.csv')
