#!/usr/bin/octave -q

if (nargin!=11)
printf("Usage: gausianexp.m <trdata> <trlabels> <tedata> <telabels> <alpha> <mina> <stepa> <maxa> <mink> <stepk> <maxk>\n")
exit(1);
end;

arg_list=argv();
trdata=arg_list{1};
trlabels=arg_list{2};
tedata=arg_list{3};
telabels=arg_list{4};
#alpha=str2num(arg_list{5});
mina=str2num(arg_list{6});
stepa=str2num(arg_list{7});
maxa=str2num(arg_list{8});

mink=str2num(arg_list{9});
stepk=str2num(arg_list{10});
maxk=str2num(arg_list{11});

load(trdata);
load(trlabels);
load(tedata);
load(telabels);

[m, Wp]=pca(X); 
printf("alpha\tk\terror\n")
for alpha=mina:stepa:maxa
for k=mink:stepk:maxk

Wtr=Wp(:,1:k)' * (X-m)'; #proyectamos a k dimensiones con pca
Wte=Wp(:,1:k)' * (Y-m)';



err=mixgaussiani(Wtr',xl,Wte',yl, 2,alpha)
printf("%d\t%d\t%d\n", alpha, k, err)

endfor
endfor

