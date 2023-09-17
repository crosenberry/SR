import xom_ann_model
import xom_rnn_model


def main():
    company = input("Please enter the name of the company to predict: Chevron or Exxon: ")
    if company != "Chevron" and company != "Exxon":
        print("Invalid company name")
        return

    network_type = input("Please enter the type of network to use: ANN or RNN: ")
    if network_type != "ANN" and network_type != "RNN":
        print("Invalid network type")
        return

    test_num = int(input("Please enter training number 1-9 or 10 for formal test: "))
    if test_num < 1 or test_num > 10:
        print("Invalid training number")
        return

    print("You entered: " + company + " " + network_type + " " + str(test_num))
    start_dates = ['2018-04-01', '2018-05-09', '2018-06-16', '2018-07-24', '2018-08-31',
                   '2018-10-08', '2018-11-15', '2018-12-23', '2019-01-30', '2019-03-09']
    end_dates = ['2018-05-10', '2018-06-17', '2018-07-25', '2018-09-01', '2018-10-09',
                 '2018-11-16', '2018-12-24', '2019-01-31', '2019-03-10', '2019-04-17']
    print("Start date of data is: " + start_dates[test_num - 1])
    print("End date of data is: " + end_dates[test_num - 1])

    if company == "Chevron" and network_type == "ANN":
        generate_chevron_ann(start_dates[test_num - 1], end_dates[test_num - 1])
    elif company == "Chevron" and network_type == "RNN":
        generate_chevron_rnn(start_dates[test_num - 1], end_dates[test_num - 1])
    elif company == "Exxon" and network_type == "ANN":
        xom_ann_model.generate_exxon_ann(start_dates[test_num - 1], end_dates[test_num - 1])
    elif company == "Exxon" and network_type == "RNN":
        xom_rnn_model.generate_exxon_rnn(start_dates[test_num - 1], end_dates[test_num - 1])


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
