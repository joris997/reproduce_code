clear;
close all;

%% symbolic variables
syms eta x t
syms b(x,t) c(t)

%% symbolic derivatives
bi = -(1/eta)*log(exp(-eta*b(x,t)) + exp(-eta*c(t)))

bi_dx = diff(bi, x)
bi_dt = diff(bi, t)