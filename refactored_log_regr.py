from LogisticRegr import *


if __name__ == '__main__':

    df = pd.read_csv('clean_data.csv', delimiter=',')

    card = df['card']  # goal parameter
    reports = df['reports']
    age = df['age']
    income = df['income']
    expenditure = df['expenditure']
    owner = df['owner']
    selfemp = df['selfemp']
    dependents = df['dependents']
    majorcards = df['majorcards']
    active = df['active']

    train_size = int(len(card) * .75)

    train_x = np.array(card[:train_size])  # goal parameter
    train_a = np.array(reports[:train_size])
    train_b = np.array(age[:train_size])
    train_c = np.array(income[:train_size])
    train_d = np.array(expenditure[:train_size])
    train_e = np.array(owner[:train_size])
    train_f = np.array(selfemp[:train_size])
    train_g = np.array(dependents[:train_size])
    train_i = np.array(majorcards[:train_size])
    train_j = np.array(active[:train_size])

    test_x = np.array(card[train_size:len(card)])  # goal parameter
    test_a = np.array(reports[train_size:len(reports)])
    test_b = np.array(age[train_size:len(age)])
    test_c = np.array(income[train_size:len(income)])
    test_d = np.array(expenditure[train_size:len(expenditure)])
    test_e = np.array(owner[train_size:len(owner)])
    test_f = np.array(selfemp[train_size:len(selfemp)])
    test_g = np.array(dependents[train_size:len(dependents)])
    test_i = np.array(majorcards[train_size:len(majorcards)])
    test_j = np.array(active[train_size:len(active)])

    dataset_sizes = {'train': train_x.shape[0], 'test': test_x.shape[0]}

    learningRate = .0001
    epochs = 50

    # Initialize model with saved weights:

    # model = LogisticRegr(9, 1)
    # model.load_state_dict(torch.load('log_regr-clean_data.pt'))
    # model.eval()

    model = LogisticRegr(9, 1)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)

    y_loss = {'train': [], 'test': []}
    y_err = {'train': [], 'test': []}
    x_epoch = []
    fig = plt.figure()
    ax0 = fig.add_subplot(121, title="Loss")
    ax1 = fig.add_subplot(122, title="Accuracy")

    def draw_curve(current_epoch):
        x_epoch.append(current_epoch)
        ax0.plot(x_epoch, y_loss['train'], 'b-', label='train', linewidth=1)
        ax0.plot(x_epoch, y_loss['test'], 'r-', label='test', linewidth=1)
        ax1.plot(x_epoch, y_err['train'], 'b-', label='train', linewidth=1)
        ax1.plot(x_epoch, y_err['test'], 'r-', label='test', linewidth=1)
        if current_epoch == 0:
            ax0.legend()
            ax1.legend()

    for epoch in range(epochs):
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train(True)
            else:
                model.train(False)

            running_loss = 0
            running_corrects = 0

            for i in range(dataset_sizes[phase]):
                if torch.cuda.is_available():
                    inputs = Variable(torch.from_numpy(np.array([train_a[i], train_b[i], train_c[i],
                                                                 train_d[i], train_e[i], train_f[i],
                                                                 train_g[i], train_i[i],
                                                                 train_j[i]])).cuda()).double()
                    labels = Variable(torch.from_numpy(np.array([train_x[i]])).cuda()).double()
                else:
                    inputs = Variable(torch.from_numpy(np.array([train_a[i], train_b[i], train_c[i],
                                                                 train_d[i], train_e[i], train_f[i],
                                                                 train_g[i], train_i[i],
                                                                 train_j[i]]))).double()
                    labels = Variable(torch.from_numpy(np.array([train_x[i]]))).double()

                optimizer.zero_grad()
                outputs = model(inputs)
                preds = round(outputs.data.item())
                loss = criterion(outputs, labels)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item()
                running_corrects += int(preds == int(labels.data.item()))

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            y_loss[phase].append(epoch_loss)
            y_err[phase].append(epoch_acc)
            if phase == 'test':
                draw_curve(epoch)

            print(f'Epoch {epoch + 1}/{epochs}:')
            print(f'{phase} Loss: {epoch_loss} Acc: {epoch_acc}\n\n')

    fig.show()
    # fig.savefig('loss_acc_plot.png')

    # torch.save(model.state_dict(), 'log_regr-clean_data.pt')
