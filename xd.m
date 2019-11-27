#!/usr/bin/octave -q




load("data/mnist/mnisttr.mat.gz");
load("data/mnist/mnisttrlabels.mat.gz");

disp("Datos cargados")

datos = X(xl < 4, :);
labels = xl(xl < 4, :);

rand("seed",23);
[N, columnas] = size(datos);
perm = randperm(N);

datos = datos(perm, :);
labels = labels(perm, :);

Ntrain = round(0.3*N);

tr = datos(1:Ntrain, :);
trlabels = labels(1:Ntrain, :);

te = datos(Ntrain+1:N, :);
telabels = labels(Ntrain+1:N,:);

for i = 0:3
	for j = i+1:3
		#i vs j
        indicesi = find(trlabels == i);
        indicesj = find(trlabels == j);
        
		datos = tr([indicesi; indicesj], :);
        
		etiquetas = trlabels([indicesi; indicesj], :);
        
		res = svmtrain(etiquetas, datos,"-q -t 1 -c 1000");

		if i == 0
		 	svm(i+j)= res;

		else
			svm(i+j+1) = res;

		endif
	endfor
endfor

disp("Svm entrenados");

for i = 1:6
	[pred, precision, _] = svmpredict(telabels,	te, svm(i),"-q");

	prediccionesVOT(:,i) = pred;
endfor

disp("Test clasificado")

for i = 1:rows(prediccionesVOT)
    for j = 1:4
        votos(i,j) = sum(prediccionesVOT(i,:) == j - 1);
    endfor
    #ganador(i,1) = max(votos(i,:));
endfor

[numvotos, ganador] = max(votos');
ganador = ganador -1;

disp("Votacion realizada")

disp("VOTACION: ")
aciertosVOT = sum(ganador' == telabels);
porcentajeAciertosVOT = aciertosVOT/rows(telabels)




for i = 1:rows(telabels)
	minimo = 0;
	maximo = 3;

	while (minimo +1 != maximo)
		if minimo == 0
			currSVM = maximo;
		else
			currSVM = minimo + maximo + 1;
		endif

		pred = svmpredict(telabels(i,:), te(i,:), svm(currSVM),"-q");

		if pred != maximo
			maximo--;
		else
			minimo++;
		endif

	endwhile
	if minimo == 0
		currSVM = maximo;
	else
		currSVM = minimo + maximo + 1;
	endif
	prediccionesDAGS(i) = svmpredict(telabels(i,:), te(i,:), svm(currSVM),"-q");

endfor

disp("DAGS: ")

aciertosDAGS = sum(prediccionesDAGS' == telabels);
porcentajeAciertosDAGS = aciertosDAGS/rows(telabels)








