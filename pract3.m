#!/usr/bin/octave -q

addpath("nnet");

#if (nargin!=2)
# printf("Usage: pract3.m nOutput nHidden");
# exit(1);
#endif
#arglist=argv();
#nOutput=str2num(arglist{1});
#nHidden=str2num(arglist{2});

nOutput = 2;
nHidden = 10;


load data/hart/tr.dat;
load data/hart/trlabels.dat;
load data/hart/ts.dat;
load data/hart/tslabels.dat;

mInput = tr';
mOutput = trlabels';
[rows, columns] = size(mOutput);
mOutputi = zeros(2,columns);
mTestInput = ts';
mTestOutput = tslabels';

for i = 1:columns
  if mOutput(i) == 1
     mOutputi(:,i) = [0,1];
     endif
  if mOutput(i) == 2  
     mOutputi(:,i) = [1,0];
     endif
  endfor

[nFeat, nSamples] = size(mInput);
nTr=floor(nSamples*0.8);
nVal=nSamples-nTr;

rand('seed',23);
indices=randperm(nSamples);

mTrainInput=mInput(:,indices(1:nTr));
mTrainOutput=mOutputi(:,indices(1:nTr));
mValiInput=mInput(:,indices((nTr+1):nSamples));
mValiOutput=mOutputi(:,indices((nTr+1):nSamples));

[mTrainInputN,cMeanInput,cStdInput] = prestd(mTrainInput);

VV.P = mValiInput;
VV.T = mValiOutput;

VV.P = trastd(VV.P,cMeanInput,cStdInput); 

MLPnet = newff(minmax(mTrainInputN),[nHidden nOutput],{"tansig","logsig"},"trainlm","","mse");

MLPnet.trainParam.show = 10;
MLPnet.trainParam.epochs = 300;

net = train(MLPnet,mTrainInputN,mTrainOutput,[],[],VV);

mTestInputN = trastd(mTestInput,cMeanInput,cStdInput);

simOut = sim(net,mTestInputN);

[rowsts, columnsts] = size(mTestInputN);

resultados = zeros(1,columnsts);

for i = 1:columnsts
  if (simOut(1,i) > simOut(2,i) && mTestOutput(i) == 2) || (simOut(1,i) < simOut(2,i) && mTestOutput(i) == 1)
    resultados(i) = 1;
  endif
endfor

sum(resultados)/columnsts


