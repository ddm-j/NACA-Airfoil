import numpy as np
from matplotlib import pyplot as plt

def get_intersect(a1, a2, b1, b2):
    """
    https://stackoverflow.com/questions/3252194/numpy-and-line-intersections
    Returns the point of intersection of the lines passing through a2,a1 and b2,b1.
    a1: [x, y] a point on the first line
    a2: [x, y] another point on the first line
    b1: [x, y] a point on the second line
    b2: [x, y] another point on the second line
    """
    s = np.vstack([a1,a2,b1,b2])        # s for stacked
    h = np.hstack((s, np.ones((4, 1)))) # h for homogeneous
    l1 = np.cross(h[0], h[1])           # get first line
    l2 = np.cross(h[2], h[3])           # get second line
    x, y, z = np.cross(l1, l2)          # point of intersection
    if z == 0:                          # lines are parallel
        return (float('inf'), float('inf'))
    return (x/z, y/z)

def rotate_vector(vec, angle):

    R = np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0, 0, 1]
    ])

    return np.matmul(R, vec)


class NACA4(object):

    def __init__(self, digits, points=50, path=None, method='linear', save=False, te=None, tepoints=None, centered=False):

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

        te_sup = ['radius', 'linear']
        if te:
            if te not in te_sup:
                raise ValueError("Trailing edge behavior {0} not supported. Options are: {1}".format(te, te_sup))
        self.te = te

        # Create Airfoil
        self.main(digits, points, path, method, save, te, tepoints, centered)

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

        if m == 0 and p == 0:
            return x, d*5*t*(0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2 + 0.2843 * x**3 - 0.1015*x**4), 0.0

        else:
            return x - d*self.thickness(x, t)*np.sin(self.theta(x, m, p)), \
                   self.camber(x, m, p) + d*self.thickness(x, t)*np.cos(self.theta(x, m, p)), 0.0


    def foil_discret(self, p, points, method='linear'):

        #points = points + 2

        if method=='linear':
            x = np.linspace(0, 1, points)
        elif method=='log':
            first = np.geomspace(1e-8, p, num=points//2)
            second = np.flip(1 - np.geomspace(0.001, 1-p, num=points//2))

            x = np.concatenate([first, second[1:]])
        else:
            raise ValueError('Input discretization method \'{0}\' not recognized. Supported '
                             'methods are \'linear\' and \'log\''.format(method))

        return x

    def trailing_edge(self, upper, lower, behavior, points=None):

        # Trailing Edge Vectors
        u = upper[-1] - upper[-2]
        mag_u = np.linalg.norm(u)
        l = lower[-1] - lower[-2]
        mag_l = np.linalg.norm(l)
        u_n = u/mag_u
        l_n = l/mag_l

        # Suggested Point Spacing for Trailing edge
        ds = (mag_u + mag_l) / 2

        if behavior == 'linear':
            # Intersection Point
            i_x, i_y = get_intersect(upper[-2, 0:2], upper[-1, 0:2], lower[-2, 0:2], lower[-1, 0:2])
            intersect = np.array([i_x, i_y, 0.0])

            # Get Segment Lengths for divisions, and number of divisions for each
            du = np.linalg.norm(intersect - upper[-1])
            nu = round(du/ds)
            dl = np.linalg.norm(intersect - lower[-1])
            nl = round(dl/ds)

            if points:
                nl = points
                nu = points

            elif nu == 0 or nl == 0:
                te_upper = intersect.reshape((1, 3))
                te_lower = intersect.reshape((1, 3))

                return te_upper, te_lower

            div_u = du/nu
            div_l = dl/nl

            if nu < 2 or nl < 2:
                raise Warning("Discretization at the trailing edge is not very good. Try increasing number of points.")

            # Create Equally Spaced Points
            mag_points_u = div_u*np.array(range(1, nu+1))
            mag_points_l = div_l*np.array(range(1, nl+1))

            te_upper = np.array([upper[-1] + u_n*i for i in mag_points_u])
            te_lower = np.array([lower[-1] + l_n*i for i in mag_points_l])

            te_upper[-1] = intersect
            te_lower[-1] = intersect

        elif behavior == 'radius':

            # Normal vectors to the tangent points
            normal_u = np.array([u_n[1], -u_n[0], u_n[2]])
            normal_l = np.array([-l_n[1], l_n[0], l_n[2]])
            pu2 = upper[-1] + 0.001*normal_u
            pl2 = lower[-1] + 0.001*normal_l

            # Get Circle Center
            c_x, c_y = get_intersect(upper[-1, 0:2], pu2[0:2], lower[-1, 0:2], pl2[0:2])
            c = np.array([c_x, c_y, 0.0])
            R = np.linalg.norm(upper[-1] - c)

            # Get Angles for Discretization
            r0 = np.array([1, 0, 0])
            alpha_u = np.arccos(np.dot(-normal_u, r0))
            alpha_l = np.arccos(np.dot(r0, -normal_l))

            # Calculate Spacing
            s_u = R*alpha_u
            nu = round(s_u / ds)
            s_l = R*alpha_l
            nl = round(s_l / ds)

            if points:
                nl = points
                nu = points

            elif nu == 0 or nl == 0:
                te_upper = (c + R*r0).reshape((1, 3))
                te_lower = (c + R*r0).reshape((1, 3))

                return te_upper, te_lower

            ds_u = s_u / nu
            dtheta_u = ds_u / R

            ds_l = s_l / nl
            dtheta_l = ds_l / R

            # Spaced Points
            R_u = -R*normal_u
            R_l = -R*normal_l

            angles_u = -dtheta_u*np.array(range(1, nu+1))
            angles_l = dtheta_l*np.array(range(1, nl+1))

            # Create Points
            te_upper = np.array([c + rotate_vector(R_u, i) for i in angles_u])
            te_lower = np.array([c + rotate_vector(R_l, i) for i in angles_l])

            # Ensure Exact Match
            te_upper[-1] = c + R*np.array([1, 0, 0])
            te_lower[-1] = c + R*np.array([1, 0, 0])

        return te_upper, te_lower

    def main(self, digits, points, path, method, save, te, tepoints, centered):
        m = float(digits[0])/100
        p = float(digits[1])/10
        t = float(digits[2:])/100
        points = int(points)

        # Create the curve discretization points
        x = self.foil_discret(p, points, method=method)

        # Calculate the upper and lower profiles
        upper = np.insert(np.array([self.foil(i, m, p, t) for i in x]), 0, np.array([0, 0, 0]), 0)
        lower = np.insert(np.array([self.foil(i, m, p, t, upper=False) for i in x]), 0, np.array([0, 0, 0]), 0)

        # Create the trailing edge discretization
        if te:
            print('Creating trailing edge. TE points: {0}'.format(tepoints))
            te_upper, te_lower = self.trailing_edge(upper, lower, behavior=te, points=tepoints)
            upper = np.concatenate((upper, te_upper), axis=0)
            lower = np.concatenate((lower, te_lower), axis=0)

        # Center the Airfoil coordinates
        if centered:
            upper[:, 0] -= 0.5
            lower[:, 0] -= 0.5

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
    parser.add_argument('--te', help='Trailing edge behavior: radius, closed. Default closed.',
                        default=None)
    parser.add_argument('--tepoints', help='Trailing edge discretization point count override.',
                        default=None)
    parser.add_argument('--centered', help='Center the airfoil such that the origin is at mid-cord.',
                        default=False)

    args = parser.parse_args()

    # Create Airfoil
    af = NACA4(**vars(args))