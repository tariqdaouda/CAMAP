import torch
import tqdm
import numpy

# ** Functions **

def save_model(model, criterion, optimizer, training_history, filename):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'criterion': criterion.state_dict(),
        'training_history': training_history
    }, filename)


def load_model(filename, empty_model, device='cpu'):
    empty_model.to(device)  # not done automatically when loading state
    state = torch.load(filename, map_location=torch.device(device))
    empty_model.load_state_dict(state['model_state_dict'])
    return empty_model


# ** Classes **

class SequenceDataset(object):
    """docstring for SequenceDataset"""
    def __init__(self, dataset, mask=None):
        super(SequenceDataset, self).__init__()

        if mask is not None:
            masked = [[], []]
            for data in dataset[0]:
                masked[0].append(data * mask)
            for data in dataset[1]:
                masked[1].append(data * mask)
            dataset = masked

        ns_ratio = len(dataset[0]) // len(dataset[1])
        samples = [dataset[0]]
        labels = [numpy.zeros(len(dataset[0]))]

        samples.extend( [dataset[1]] * int(ns_ratio) )
        labels.extend( [numpy.ones(len(dataset[1]))] * int(ns_ratio) )

        samples = numpy.concatenate(samples, 0)
        labels = numpy.concatenate(labels, 0)

        self.samples = torch.tensor(samples, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]


class Classifier(torch.nn.Module):
    def __init__(self, input_len, bias=False):
        super(Classifier, self).__init__()
        self.embeds = torch.nn.Embedding(65, 2)
        weight = self.embeds.weight.clone().detach()
        weight[0] = torch.tensor([0, 0], dtype=torch.float32, requires_grad=False)
        weight = torch.nn.Parameter(weight, requires_grad=True)
        self.embeds.weight = weight
        self.out_layer = torch.nn.Linear(input_len*2, 2, bias=bias)
        self.probas = torch.nn.Softmax(dim=1)
        self.input_len = input_len

    def _assert_type(self, sequences):
        device = self.state_dict()['embeds.weight'].device.type
        try:
            if sequences.device.type != device:
                sequences = sequences.to(device)
        except AttributeError:
            sequences = torch.tensor(sequences, dtype=torch.long)
            sequences = sequences.to(device)
        return sequences

    def forward(self, sequences):
        out = self.embeds(sequences)
        out = out.view(-1, 2*self.input_len)
        out = self.out_layer(out)
        return out

    def get_score(self, sequences):
        sequences = self._assert_type(sequences)
        out = self(sequences)
        out = self.probas(out)
        return out


class Trainer(object):
    """docstring for Trainer"""
    def __init__(self, net, lr, device, optimizer):
        super(Trainer, self).__init__()
        self.net = net
        self.criterion = torch.nn.CrossEntropyLoss()
        opt_switch = {
            "adam": torch.optim.Adam(net.parameters(), lr=lr),
            "adagrad": torch.optim.Adagrad(net.parameters(), lr=lr),
            "sgd": torch.optim.SGD(net.parameters(), lr=lr, momentum=0.)
        }
        self.optimizer = opt_switch[optimizer]

        self.device = device
        self.history = {}

    def add_history(self, setname, value, autosave=True):
        if setname not in self.history:
            self.history[setname] = {
                "best": value,
                "curve": [value]
            }
            filename = self.get_best_model_filename(setname)
            self.save(filename)
        else:
            self.history[setname]["curve"].append(value)
            if value < self.history[setname]["best"]:
                self.history[setname]["best"] = value
                filename = self.get_best_model_filename(setname)
                self.save(filename)
        return False

    def get_best_model_filename(self, setname):
        return "%s-bestMin-score.pytorch" % setname

    def load_best_model(self, setname):
        filename = self.get_best_model_filename(setname)
        return load_model(filename, self.net)

    def save(self, filename):
        save_model(self.net, self.criterion, self.optimizer, self.history, filename)

    def train(self, epochs, traindata, testdata, validationdata,
              mini_batch_size=256, mask=None, nb_data_workers=8):
        trainloader = torch.utils.data.DataLoader(
            SequenceDataset(traindata, mask),
            batch_size=mini_batch_size,
            shuffle=True,
            num_workers=nb_data_workers
        )
        testloader = torch.utils.data.DataLoader(
            SequenceDataset(testdata, mask),
            batch_size=len(SequenceDataset(testdata)),
            shuffle=False,
            num_workers=nb_data_workers
        )
        validationloader = torch.utils.data.DataLoader(
            SequenceDataset(validationdata, mask),
            batch_size=len(SequenceDataset(testdata)),
            shuffle=False,
            num_workers=nb_data_workers
        )

        pbar = tqdm.trange(epochs)

        self.net.train()
        for epoch in pbar:  # loop over the dataset multiple times
            running_loss = 0.0
            for data in trainloader:
                running_loss += self.train_pass(data)
            self.add_history("train", running_loss)

            for data in testloader:
                test_loss = self.test_pass(data)
                self.add_history("test", test_loss)

            for data in validationloader:
                validation_loss = self.test_pass(data)
                self.add_history("validation", validation_loss)

            pbar.set_description("%.5f" % running_loss)

        print('done')

    def _get_input_data(self, data):
        samples, labels = data
        try:
            samples = samples.to(self.device)
            labels = labels.to(self.device)
        except AttributeError:
            samples = torch.tensor(samples, dtype=torch.long)
            samples = samples.to(self.device)
            labels = torch.tensor(labels, dtype=torch.long)
            labels = labels.to(self.device)
        return samples, labels

    def train_pass(self, data):
        samples, labels = self._get_input_data(data)
        self.optimizer.zero_grad()

        outputs = self.net(samples)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def test_pass(self, data):
        self.net.eval()
        with torch.no_grad():
            samples, labels = self._get_input_data(data)
            outputs = self.net(samples)
            loss = self.criterion(outputs, labels)
            return loss.item()

    def predict_pass(self, data):
        self.net.eval()
        with torch.no_grad():
            samples, labels = self._get_input_data(data)
            outputs = self.net.get_score(samples)
            outputs_0, outputs_1 = list(zip(*outputs))
            _, predicted = torch.max(outputs.data, 1)
            return {
                "predictions": predicted,
                "proba_scores_0": outputs_0,
                "proba_scores_1": outputs_1,
                "targets": labels
            }

    def test(self, testdata, mask=None, nb_data_workers=8) :
        testloader = torch.utils.data.DataLoader(
            SequenceDataset(testdata, mask),
            batch_size=64,
            shuffle=False,
            num_workers=nb_data_workers)
        for data in testloader:
            self.test_pass(data)

    def predict(self, testdata, mask=None, nb_data_workers=8):
        testloader = torch.utils.data.DataLoader(
            SequenceDataset(testdata, mask),
            batch_size=64,
            shuffle=False,
            num_workers=nb_data_workers)

        res = {name: [] for name in ["predictions", "targets", "proba_scores_0", "proba_scores_1"]}
        for data in testloader:
            out = self.predict_pass(data)
            for name in res: res[name].extend(out[name])
        for name in res: res[name] = numpy.array(res[name])

        res["accuracy"] = (res["predictions"] == res["targets"]).sum() / len(res["predictions"])

        return res
