import xom_ann_model
import xom_rnn_model
import cvx_ann_model
import cvx_rnn_model


# This program is the main program that will be used to run the ANN and RNN models
def main():
    global i
    company = input("Please enter the name of the company to predict: Chevron or Exxon: ")
    if company != "Chevron" and company != "Exxon":
        print("Invalid company name")
        return

    network_type = input("Please enter the type of network to use: ANN or RNN: ")
    if network_type != "ANN" and network_type != "RNN":
        print("Invalid network type")
        return

    print("You entered: " + company + " " + network_type)
    start_dates = '2018-04-01'
    end_dates = '2019-05-05'
    print("Start date of data is: " + start_dates)
    print("End date of data is: " + end_dates)

    seed_values = [44, 17, 4, 23, 87, 194, 222, 321, 101, 405, 784]

    if company == "Chevron" and network_type == "ANN":
        for i in range(len(seed_values)):
            cvx_ann_model.generate_chevron_ann(start_dates, end_dates, seed_values[i])
    elif company == "Chevron" and network_type == "RNN":
        for i in range(len(seed_values)):
            cvx_rnn_model.generate_chevron_rnn(start_dates, end_dates, seed_values[i])
    elif company == "Exxon" and network_type == "ANN":
        for i in range(len(seed_values)):
            xom_ann_model.generate_exxon_ann(start_dates, end_dates, seed_values[i])
    elif company == "Exxon" and network_type == "RNN":
        for i in range(len(seed_values)):
            xom_rnn_model.generate_exxon_rnn(start_dates, end_dates, seed_values[i])


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
