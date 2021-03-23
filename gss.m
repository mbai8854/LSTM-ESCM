function sol = gss(f,a,b,tol)
if (nargin < 1)
    a = -4;
    b = 4;
    tol = 1e-5;
    f = @(x) ((x.^3-x).^2-2*x.^3).^2;
end
if (nargin < 4)
    tol = 1e-5;
end
invphi = (sqrt(5) - 1) / 2;% 1/phi
invphi2 = (3 - sqrt(5)) / 2;% 1/phi^2

temp = a;
a = min(temp,b);
b = max(temp,b);
h = b - a;
if h <= tol
    sol = mean([a,b]);
else
    % required steps to achieve tolerance
    n = ceil(log(tol/h)/log(invphi));
    c = a + invphi2 * h;
    d = a + invphi * h;
    yc = f(c);
    yd = f(d);
    
    for k =1:n-1
        if yc < yd
            b = d;
            d = c;
            yd = yc;
            h = invphi*h;
            c = a + invphi2 * h;
            yc = f(c);
        else
            a = c;
            c = d;
            yc = yd;
            h = invphi*h;
            d = a + invphi * h;
            yd = f(d);
        end
    end
    
    if yc < yd
        sol = mean([a,d]);
    else
        sol = mean([c,b]);
    end
end
end







