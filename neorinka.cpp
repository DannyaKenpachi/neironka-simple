#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>

using namespace std;

class Layer {
public:
    vector<double> inputs;       
    vector<double> weights;
    double sdvig;

    Layer(int input_size) {
        inputs.resize(input_size);
    }

    double sigmoid(double x) {
        return 1.0 / (1.0 + exp(-x));
    }

    double sigmoid_derivative(double x) {
        return (1.0 - sigmoid(x)) * sigmoid(x);
    }

    void SetWeights_Bias() {
        weights.resize(inputs.size());
        for (int i; i < weights.size(); ++i) {
            weights[i] = (double)rand() / RAND_MAX;
        }
        sdvig = (double)rand() / RAND_MAX;
    }

    double prymoe_raspredelenie(vector<double> inputs) {
        double summa = 0;
        for (int i = 0; i < weights.size(); ++i) {
            summa += weights[i] * inputs[i];
        }
        return sigmoid(summa + sdvig);
    }

    void obychenie (vector<double> inputs, double itog, double u_obychenie = 0.01, double porog_ochibke = 1e-3) {
        cout << "Процесс обучения:" << endl;
        int level = 0;
        bool f = true;
        while (f) {
            double output = prymoe_raspredelenie(inputs);
            double loss = fabs(itog - output);
            
            if (loss <= porog_ochibke) {
                cout << "Обучение завершено" << endl << "Уровень - " << level << " Ошибка - " << loss << endl;
                f = false;
            }

            if ( level % 10 == 0){
                cout << "Эпоха " << level << ", Выходные данные: " << output << ", Ошибка: " << loss << endl;
            }

            double summa = sdvig;
            for (int i = 0; i < weights.size(); ++i) {
                summa += weights[i] * inputs[i];
            }

            double derivative = sigmoid_derivative(summa);

            for (int i = 0; i < weights.size(); ++i) {
                double gradient = (itog > output ? 1 : -1) * inputs[i] * derivative;
                weights[i] += u_obychenie * gradient;
            }
            sdvig += u_obychenie * sigmoid_derivative(sdvig);
            level += 1;
        }
    }
};

int main() {
    cout << "Простая Нейронная сеть" << endl;
    Layer nn(3);
    cout << "Количество входных данных = " << 3 << endl;
    nn.inputs = {1.0, 0.0, 1.0};
    cout << "Входные данные: 1, 0, 1" << endl;
    nn.SetWeights_Bias();
    double itog = 0.1;
    nn.obychenie(nn.inputs, itog);
    return 0;
}