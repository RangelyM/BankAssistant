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

    model = LogisticRegr(9, 1)

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)

    y_loss = {'train': [], 'test': []}
    y_err = {'train': [], 'test': []}
    x_epoch = []
    fig = plt.figure()
    ax0 = fig.add_subplot(121, title="loss")
    ax1 = fig.add_subplot(122, title="top1err")

    def draw_curve(current_epoch):
        x_epoch.append(current_epoch)
        ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')
        ax0.plot(x_epoch, y_loss['test'], 'ro-', label='test')
        ax1.plot(x_epoch, y_err['train'], 'bo-', label='train')
        ax1.plot(x_epoch, y_err['test'], 'ro-', label='test')
        if current_epoch == 0:
            ax0.legend()
            ax1.legend()
        fig.show()

    running_loss = 0.
    running_corrects = 0.

    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}\n', '-' * 20)

        for i in range(len(train_x)):
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
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        x_epoch.append(epoch)
        running_loss += loss.item()
        running_corrects += float(torch.sum(preds == labels.data))
        print(f'epoch {epoch}, loss {loss.item()}')

    epoch_loss = running_loss / dataset_sizes['train']
    epoch_acc = running_corrects / dataset_sizes['train']

    print(f'Train:\n\nLoss: {epoch_loss}')

    predicted = []
    for epoch in range(len(test_x)):
        with torch.no_grad():
            if torch.cuda.is_available():
                predicted.append(
                    model(Variable(torch.from_numpy(np.array([train_a[epoch], train_b[epoch], train_c[epoch],
                                                              train_d[epoch], train_e[epoch], train_f[epoch],
                                                              train_g[epoch], train_i[epoch],
                                                              train_j[epoch]])).cuda()).double()).cpu().data.numpy())
            else:
                predicted.append(
                    model(Variable(torch.from_numpy(np.array([train_a[epoch], train_b[epoch], train_c[epoch],
                                                              train_d[epoch], train_e[epoch], train_f[epoch],
                                                              train_g[epoch], train_i[epoch],
                                                              train_j[epoch]]))).double()).data.numpy())
    print(predicted)

    print(*model.parameters(), sep='\n\n')
