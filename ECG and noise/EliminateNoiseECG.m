clc,clearvars,close all
% Programa que filtra una senal de ECG por medio de backpropagation SIN
% retroalimentación
% ECG1 es la senal obtenida de la base de datos de Physionet
% ECG2 es la senal obtenida de un circuito
% Cargar las senales
load('rec_1m.mat');
ECG2 = val(1:end);
fs = 500;
amp = (max(ECG2)- min(ECG2))/20e-3;
ECG2 = (ECG2/amp)*500;
ECG2 = ECG2';
t = 0:1/fs:(length(ECG2)-1)/fs;
noise = 20*sin(2 * pi * 60 * t)/10;
ECG2 = noise' + ECG2;
figure, subplot 311, plot(ECG2), hold on

ECG1 = fopen('rec_1.dat');
ECG1 = double(fread(ECG1, 'int'));
amp = (max(ECG1)- min(ECG1))/20e-3;
ECG1 = (ECG1/amp)*500;
ECG1 = ECG1(1:length(ECG2));
plot(ECG1), legend('ECG c/ruido', 'ECG sin ruido'), title('Señales Originales')

% Declaración de parámetros iniciales
W = rand(1,3);
b = rand(1);
alpha = 0.01;

%% Primer bloque para obtener el ruido de la senal de ECG2
patron = zeros(3,1);
ruido = zeros(1,length(ECG1));
for t = 1:length(ECG1)
    if t == 1
        patron(1) = ECG1(t);
    elseif t==2
        patron(1) = ECG1(t);
        patron(2) = ECG1(t-1);
    else
        patron(1) = ECG1(t);
        patron(2) = ECG1(t-1);
        patron(3) = ECG1(t-2);
    end
    a = purelin((W*patron) + b);
    error = ECG2(t) - a;
    ruido(t) = error;
    W = W + (2*alpha*error*patron');
    b = b + 2*alpha*error;
end
subplot 312, plot(ruido), title('Ruido')
ruido = ruido';


%% Segundo bloque para obtener la senal ECG2 filtrada
W2 = rand(1,3);
b2 = rand(1);
alpha = 0.001;
patron2 = zeros(3,1);
ECG = zeros(1,length(ruido));
for t = 1:length(ruido)
    if t == 1
        patron2(1) = ruido(t);
    elseif t==2
        patron2(1) = ruido(t);
        patron2(2) = ruido(t-1);
    else
        patron2(1) = ruido(t);
        patron2(2) = ruido(t-1);
        patron2(3) = ruido(t-2);
    end
    a2 = purelin((W2*patron2) + b2);
    error2 = ECG1(t) - a2;
    ECG(t) = error2;
    W2 = W2 + (2*alpha*error2*patron2');
    b2 = b2 + 2*alpha*error2;
end
subplot 313, plot(ECG), title('Señal de ECG filtrada'), hold on
load('rec_1m.mat');
ECG2 = val(1:end);
fs = 500;
amp = (max(ECG2)- min(ECG2))/20e-3;
ECG2 = (ECG2/amp)*500;
ECG2 = ECG2';
plot(ECG2), ylim([-10 10])