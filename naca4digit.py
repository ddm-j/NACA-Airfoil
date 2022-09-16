import numpy as np
from matplotlib import pyplot as plt

class NACA4(object):

    def __init__(self, digits, points=50, path=None, method='linear', save=False):

        # Class Attributes
        self.n_points = 0
        self.method = method
        self.digits = digits
        self.p = 0
        self.m = 0
        self.t = 0
        self.upper = None
        self.lower = None
        self.ordered_points = None

        # Create Airfoil
        self.main(digits, points, path, method, save)

    def theta(self, x, m, p):

        if x <= p:
            return np.arctan(((2*m)/p**2)*(p-x))
        elif x > p:
            return np.arctan((2*m/(1-p)**2)*(p-x))


    def camber(self, x, m, p):

        if x <= p:
            return (m/p**2)*(2*p*x - x**2)
        elif x > p:
            return (m/(1-p)**2)*((1-2*p)+2*p*x-x**2)


    def thickness(self, x, t):

        return 5*t*(0.2969*np.sqrt(x) - 0.126*x - 0.3516*x**2 + 0.2843*x**3 - 0.1015*x**4)


    def foil(self, x, m, p, t, upper=True):

        d = 1 if upper else -1

        return x - d*self.thickness(x, t)*np.sin(self.theta(x, m, p)), \
               self.camber(x, m, p) + d*self.thickness(x, t)*np.cos(self.theta(x, m, p)), 0.0


    def foil_discret(self, p, points, method='linear'):

        if method=='linear':
            x = np.linspace(0, 1, points)
        elif method=='log':
            first = np.geomspace(0.001, p, num=points//2)
            second = np.flip(1 - np.geomspace(0.001, 1-p, num=points//2))

            x = np.concatenate([first, second[1:]])
        else:
            raise ValueError('Input discretization method \'{0}\' not recognized. Supported '
                             'methods are \'linear\' and \'log\''.format(method))

        return x


    def main(self, digits, points, path, method, save):
        m = float(digits[0])/100
        p = float(digits[1])/10
        t = float(digits[2:])/100
        points = int(points)

        # Create the curve discretization points
        x = self.foil_discret(p, points, method=method)

        # Calculate the upper and lower profiles
        upper = np.insert(np.array([self.foil(i, m, p, t) for i in x]), 0, np.array([0, 0, 0]), 0)
        lower = np.insert(np.array([self.foil(i, m, p, t, upper=False) for i in x]), 0, np.array([0, 0, 0]), 0)

        # Save the files
        if save:
            if path:
                np.savetxt(path+'/{0}_upper.txt'.format(digits), upper, delimiter=' ')
                np.savetxt(path+'/{0}_lower.txt'.format(digits), lower, delimiter=' ')
            else:
                np.savetxt('{0}_upper.txt'.format(digits), upper, delimiter=' ')
                np.savetxt('{0}_lower.txt'.format(digits), lower, delimiter=' ')

        # Add to class
        self.m, self.p, self.t = m, p, t
        self.upper, self.lower = upper, lower
        self.n_points = len(upper)+len(lower)
        self.ordered_points = np.concatenate([np.flipud(upper[1:]), lower])

if __name__ == "__main__":
    import argparse

    # File Inputs
    parser = argparse.ArgumentParser(
        description='Generate cambered NACA 4 digit airfoil profile points.')
    parser.add_argument('digits', help='4 digits of the NACA Profile.')
    parser.add_argument('--points', help='Number of points to use for the airfoil sampling. Default = 100',
                        default=100)
    parser.add_argument('--path', help='Output path for saving the profile.')
    parser.add_argument('--method', help='Method for discretization of the airfoil surface: linear or log.',
                        default='linear')
    parser.add_argument('--save', help='Option to save the airfoil coordinates as TXT.',
                        default=True)

    args = parser.parse_args()

    # Create Airfoil
    af = NACA4(**vars(args))