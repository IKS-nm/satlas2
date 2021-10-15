from satlas2.core import Model, Parameter

import numpy as np

__all__ = ['ToFICR']

def ExtractionArray():
    a = 6.99797
    b = 6.9225642
    c = 0.0015824
    d = -0.0000079
    e = 3.2265697
    f = 0.0373329
    g = -0.0000951
    h = 10.0476155
    i = -0.0024371
    j = -0.0000381
    k = 26.6285599
    l = -0.090522
    m = 0.0000787
    n = 11.2570807
    o = -0.0326127
    p = 0.0000241
    q = 3.0220748
    r = -0.0069769
    s = 0.0000041
    
    array_b_e = np.array([
        [0,  2,  0.4,  a,  0,  0],
        [2,  4,  -0.68,  a,  0,  0],
        [4,  6,  -1.5,  a,  0,  0],
        [6,  8,  -2.1,  a,  0,  0],
        [8,  10,  -2.6,  a,  0,  0],
        [10,  15,  -3.2,  a,  0,  0],
        [15,  20,  -3.8,  a,  0,  0],
        [20,  30,  -4.2,  a,  0,  0],
        [30,  40,  -4.3,  a,  0,  0],
        [40,  60,  -4.4,  a,  0,  0],
        [60,  200,  -4.4,  b,  c,  d],
        [200,  230,  -4.4,  e,  f,  g],
        [230,  270,  -4.9,  e,  f,  g],
        [270,  300,  -5.4,  e,  f,  g],
        [300,  340,  -5.9,  h,  i,  j],
        [340,  370,  -6.5,  h,  i,  j],
        [370,  400,  -16.8,  h,  i,  j],
        [400,  440,  -180.9,  k,  l,  m],
        [440,  480,  -334.2,  k,  l,  m],
        [480,  510,  -271.5,  k,  l,  m],
        [510,  540,  -204.8,  k,  l,  m],
        [540,  600,  -836.7,  n,  o,  p],
        [600,  640,  -1500,  n,  o,  p],
        [640,  740,  -15300,  q,  r,  s],
        [740,  840,  -30000,  q,  r,  s],
        [840,  1600,  -30000,  0.06,  0,  0],
    ])
    return array_b_e

class ToFICR(Model):
    def __init__(self, f1=0, f2=0, f3=0, AxE=10, p=0.1, rm=1.413, rp=0.1, phase=0, mass=85, conversions=1.0, Ttot=400, Tf1=100, Tf2=300, ratio=25, ratio2=0, bkg=0.1, bkgratio=0, charge=1.0, profile="Konig", name=None):
        super().__init__(name=name)

        self.params = {'f1': Parameter(value=f1, vary=True),
                       'f2': Parameter(value=f2, vary=False),
                       'f3': Parameter(value=f3, vary=False),
                       'AxE': Parameter(value=AxE, vary=True),
                       'p': Parameter(value=p, vary=False),
                       'rm': Parameter(value=rm, vary=True),
                       'rp': Parameter(value=rp, vary=True),
                       'phase': Parameter(value=phase, vary=True),
                       'mass': Parameter(value=mass, vary=True),
                       'conversions': Parameter(value=conversions, vary=False),
                       'Ttot': Parameter(value=Ttot, vary=False),
                       'Tf1': Parameter(value=Tf1, vary=False),
                       'Tf2': Parameter(value=Tf2, vary=False),
                       'ratio': Parameter(value=ratio, vary=True, min=0, max=100),
                       'ratio2': Parameter(value=ratio2, vary=False, min=0, max=100),
                       'bkg': Parameter(value=bkg, vary=False),
                       'bkgratio': Parameter(value=bkgratio, vary=False, min=0, max=100),
                       'charge': Parameter(value=charge, vary=False),
                }
        self.mapping = {'Konig': self.konig,
                        'Double Konig': self.doublekonig,
                        'Triple Konig': self.triplekonig,
                        'Ideal Ramsey': self.ramsey,
                        'Damped Ramsey': self.dampedramsey,
                        'Asymmetrical Ramsey': self.asymmetricramsey}
        self.profiles = list(self.mapping.keys())
        self.profile = profile


    def _getTofFasterLoop(self, mu):
        q = self.params['charge'].value
        mass = self.params['mass'].value
        axial_energy = self.params['AxE'].value

        field_points = ExtractionArray()

        total_energy = axial_energy+q*field_points[0][2]+mu*field_points[0][3]

        EPSILON = 1.0e-30
        mu = np.array(mu)
        this_tof = np.zeros((len(field_points), mu.shape[0]))

        for k, i in enumerate(field_points):
            x1 = i[0]
            x2 = i[1]
            a = -mu*i[5]
            b = -mu*i[4]
            c = total_energy - q*i[2] - mu*i[3]
            determinant = b*b - 4 * a * c

            mask = np.logical_and.reduce([np.abs(a) > EPSILON, np.abs(b) > EPSILON])
            mask_posa_posdisc = np.logical_and.reduce([mask, a > 0, determinant > 0])
            mask_posa_negdisc = np.logical_and.reduce([mask, a > 0, determinant < 0])
            mask_nega_posdisc = np.logical_and.reduce([mask, a < 0, determinant > 0])
            mask_2 = np.logical_and.reduce([np.abs(a) <= EPSILON, np.abs(b) > EPSILON])
            mask_3 = np.logical_and.reduce([np.abs(a) <= EPSILON, np.abs(b) <= EPSILON])

            m = mask_posa_posdisc
            if m.sum() > 0:
                this_tof[k][m] += 1/np.sqrt(a[m]) * (    #Schaum 17.13.1
                              np.log(2 * np.sqrt(a[m]) * np.sqrt(a[m] * np.power(x2,2) + b[m] * x2 + c[m]) + 2 * a[m] * x2 + b[m])
                             -np.log(2 * np.sqrt(a[m]) * np.sqrt(a[m] * np.power(x1,2) + b[m] * x1 + c[m]) + 2 * a[m] * x1 + b[m])
                            )
            m = mask_posa_negdisc
            if m.sum() > 0:
                this_tof[k][m] = 1 / np.sqrt(a[m]) * (
                              np.arcsinh((2 * a[m] * x2 + b[m]) / np.sqrt(4 * a[m] * c[m] - b[m] * b[m]))
                             -np.arcsinh((2 * a[m] * x1 + b[m]) / np.sqrt(4 * a[m] * c[m] - b[m] * b[m]))
                              )
            m = mask_nega_posdisc
            if m.sum() > 0:
                this_tof[k][m] = - 1 / np.sqrt(-a) * (
                                  np.arcsin((2 * a[m] * x2 + b[m]) / np.sqrt(determinant[m]))
                                 -np.arcsin((2 * a[m] * x1 + b[m]) / np.sqrt(determinant[m]))
                                )
            m = mask_2
            if m.sum() > 0:
                this_tof[k][m] = 2 / np.sqrt(b[m]) * (
                              np.sqrt(b[m] * x2 + c[m])
                             -np.sqrt(b[m] * x1 + c[m])
                            )
            m = mask_3
            if m.sum() > 0:
                this_tof[k][m] = 1 / np.sqrt(c[m]) * (x2 - x1)
        time_of_flight = this_tof.sum(axis=0)
        return time_of_flight * np.sqrt(mass / 2) / 10

    def _getTofFaster(self, mu):
        q = self.params['charge'].value
        mass = self.params['mass'].value
        axial_energy = self.params['AxE'].value

        field_points = ExtractionArray()

        total_energy = axial_energy+q*field_points[0][2]+mu*field_points[0][3]

        EPSILON = 1.0e-30
        mu = np.array(mu)
        this_tof = np.zeros((len(field_points), mu.shape[0]))

        mu.shape = (1, mu.shape[0])
        x1 = np.ones(mu.shape) * np.atleast_2d(field_points[:, 0]).T
        x2 = np.ones(mu.shape) * np.atleast_2d(field_points[:, 1]).T
        a = -mu*np.atleast_2d(field_points[:, 5]).T
        b = -mu*np.atleast_2d(field_points[:, 4]).T

        c = total_energy - q*np.atleast_2d(field_points[:, 2]).T - mu*np.atleast_2d(field_points[:, 3]).T

        determinant = b*b - 4 * a * c

        mask = np.logical_and.reduce([np.abs(a) > EPSILON, np.abs(b) > EPSILON])
        mask_posa_posdisc = np.logical_and.reduce([mask, a > 0, determinant > 0])
        mask_posa_negdisc = np.logical_and.reduce([mask, a > 0, determinant < 0])
        mask_nega_posdisc = np.logical_and.reduce([mask, a < 0, determinant > 0])
        mask_2 = np.logical_and.reduce([np.abs(a) <= EPSILON, np.abs(b) > EPSILON])
        mask_3 = np.logical_and.reduce([np.abs(a) <= EPSILON, np.abs(b) <= EPSILON])

        m = mask_posa_posdisc
        if m.sum() > 0:
            this_tof[m] = 1/np.sqrt(a[m]) * (    #Schaum 17.13.1
                          np.log(2 * np.sqrt(a[m]) * np.sqrt(a[m] * np.power(x2[m],2) + b[m] * x2[m] + c[m]) + 2 * a[m] * x2[m] + b[m])
                         -np.log(2 * np.sqrt(a[m]) * np.sqrt(a[m] * np.power(x1[m],2) + b[m] * x1[m] + c[m]) + 2 * a[m] * x1[m] + b[m])
                        )
        m = mask_posa_negdisc
        if m.sum() > 0:
            this_tof[m] = 1 / np.sqrt(a[m]) * (
                          np.arcsinh((2 * a[m] * x2[m] + b[m]) / np.sqrt(4 * a[m] * c[m] - b[m] * b[m]))
                         -np.arcsinh((2 * a[m] * x1[m] + b[m]) / np.sqrt(4 * a[m] * c[m] - b[m] * b[m]))
                          )
        m = mask_nega_posdisc
        if m.sum() > 0:
            this_tof[m] = - 1 / np.sqrt(-a[m]) * (
                              np.arcsin((2 * a[m] * x2[m] + b[m]) / np.sqrt(determinant[m]))
                             -np.arcsin((2 * a[m] * x1[m] + b[m]) / np.sqrt(determinant[m]))
                            )
        m = mask_2
        if m.sum() > 0:
            this_tof[m] = 2 / np.sqrt(b[m]) * (
                          np.sqrt(b[m] * x2[m] + c[m])
                         -np.sqrt(b[m] * x1[m] + c[m])
                        )
        m = mask_3
        if m.sum() > 0:
            this_tof[m] = 1 / np.sqrt(c[m]) * (x2[m] - x1[m])
        time_of_flight = this_tof.sum(axis=0)
        return time_of_flight * np.sqrt(mass / 2) / 10

    def _magneticMoment(self, x, peak=1):
        conversions = self.params['conversions'].value
        time_tot = self.params['Ttot'].value*1e-3
        nu_c = self.params['f{:.0f}'.format(peak)].value
        mass = self.params['mass'].value
        pressure = self.params['p'].value*1e-7
        rad_minus0 = self.params['rm'].value*1e-3
        rad_plus0 = self.params['rp'].value*1e-3
        phase = self.params['phase'].value / 360 * 2 * np.pi

        # Phase is in radians
        omega_c = nu_c*2.*np.pi
        omega_rf = x*2.*np.pi

        omega_minus = 1700*2*np.pi
        omega_plus = (omega_c-omega_minus)

        k0 = conversions / time_tot *np.pi
        delta = pressure * omega_c / (1000 * 7 * 18.8e-4)
        gamma = delta / (omega_plus - omega_minus)

        omega_b1 = 0.5*np.sqrt(np.power((omega_rf-omega_c),2)+np.power((gamma*omega_c-k0),2))
        omega_b2 = 0.5*np.sqrt(np.power((omega_rf-omega_c),2)+np.power((gamma*omega_c+k0),2))
        omega_b = np.sqrt(omega_b1*omega_b2)

        theta1 = np.arctan((omega_rf-omega_c)/(gamma*omega_c-k0))
        theta1 = theta1*(theta1 >= 0) + (theta1+np.pi)*(theta1 < 0)

        theta2 = np.arctan((omega_rf-omega_c)/(gamma*omega_c+k0))
        theta2 = theta2*(theta2 >= 0) + (theta2+np.pi)*(theta2 < 0)

        theta = 0.5*(theta1+theta2)
        x = omega_b*time_tot
        y = 0.5*(omega_rf-omega_c)
        l = np.cosh(x*np.cos(theta))*np.cos(x*np.sin(theta)) #Re[np.cosh(xe^(i*theta))]
        m = np.sinh(x*np.cos(theta))*np.sin(x*np.sin(theta)) #Im[np.cosh(xe^(i*theta))]
        n = np.sinh(x*np.cos(theta))*np.cos(x*np.sin(theta)) #Re[np.sinh(xe^(i*theta))]
        p = np.cosh(x*np.cos(theta))*np.sin(x*np.sin(theta)) #Im[np.sinh(xe^(i*theta))]
        q1 = 0.5/omega_b*(rad_plus0 *(gamma*omega_c*np.cos(theta)+2.0*y*np.sin(theta))+rad_minus0*k0*(np.cos(phase)*np.cos(theta)+np.sin(phase)*np.sin(theta)))
        q2 = 0.5/omega_b*(rad_minus0*(gamma*omega_c*np.cos(theta)+2.0*y*np.sin(theta))+rad_plus0 *k0*(np.cos(phase)*np.cos(theta)-np.sin(phase)*np.sin(theta)))

        r1 = 0.5/omega_b*(rad_plus0 *(-gamma*omega_c*np.sin(theta)+2.0*y*np.cos(theta))+rad_minus0*k0*(-np.cos(phase)*np.sin(theta)+np.sin(phase)*np.cos(theta)))
        r2 = 0.5/omega_b*(rad_minus0*(-gamma*omega_c*np.sin(theta)+2.0*y*np.cos(theta))+ rad_plus0*k0*(-np.cos(phase)*np.sin(theta)-np.sin(phase)*np.cos(theta)))

        rad_plus_RE = np.exp(-delta*time_tot*0.5)*(np.cos(y*time_tot)*(rad_plus0 *l-(q1*n-r1*p))-np.sin(y*time_tot)*(rad_plus0 *m-(r1*n+p*q1)))
        rad_minus_RE = np.exp(-delta*time_tot*0.5)*(np.cos(y*time_tot)*(rad_minus0*l+(q2*n-r2*p))-np.sin(y*time_tot)*(rad_minus0*m+(r2*n+p*q2)))

        rad_plus_IM = np.exp(-delta*time_tot*0.5)*(np.sin(y*time_tot)*(rad_plus0 *l-(q1*n-r1*p))+np.cos(y*time_tot)*(rad_plus0 *m-(r1*n+p*q1)))
        rad_minus_IM = np.exp(-delta*time_tot*0.5)*(np.sin(y*time_tot)*(rad_minus0*l+(q2*n-r2*p))+np.cos(y*time_tot)*(rad_minus0*m+(r2*n+p*q2)))

        rad_minus = np.sqrt(np.power(rad_minus_RE,2.0) + np.power(rad_minus_IM,2.0))
        rad_plus = np.sqrt(np.power(rad_plus_RE, 2.0) + np.power(rad_plus_IM, 2.0))
        energy_minus = 0.5 * mass * 1.660520e-27 / 1.602177e-19 * np.power((rad_minus * omega_minus), 2.0)
        energy_plus = 0.5 * mass * 1.660520e-27 / 1.602177e-19 * np.power((rad_plus * omega_plus), 2.0)
        energy = energy_minus + energy_plus
        return energy #Returns radial energy in eV

    def _ramsey(self, x, peak=1):
        conversions = self.params['conversions'].value
        time_fringe = self.params['Tf1'].value * 1e-3
        time_tot = self.params['Ttot'].value * 1e-3
        nu_c = self.params['f{:.0f}'.format(peak)].value
        mass = self.params['mass'].value
        rad_minus0 = self.params['rm'].value * 1e-3

        time_wait = time_tot-2.*time_fringe
        omega_c =  nu_c*2.*np.pi
        omega_rf = x*2.*np.pi
        delta = omega_rf-omega_c
        g = conversions /time_fringe*np.pi*250./1000.
        omega_R = np.sqrt(np.power(2.*g,2.0)+np.power(delta,2.0))
        conversion = np.power(2.*g/omega_R *np.sin(omega_R*time_fringe/2.),2.0)*np.power(2.*np.cos(delta*time_wait/2.)*np.cos(omega_R*time_fringe/2.)-2.*delta/omega_R*np.sin(delta*time_wait/2.)*np.sin(omega_R*time_fringe/2.),2.0)
        omega_plus = omega_c
        energy_plus  = conversion*0.5*mass*1.660520e-27/1.602177e-19*np.power((rad_minus0*omega_plus),2.)
        return energy_plus

    def _ramseydamp(self, x, peak=1):
        conversions = self.params['conversions'].value
        time_fringe = self.params['Tf1'].value * 1e-3
        time_tot = self.params['Ttot'].value * 1e-3
        nu_c = self.params['f{:.0f}'.format(peak)].value
        mass = self.params['mass'].value
        pressure = self.params['p'].value * 1e-7
        rad_minus0 = self.params['rm'].value * 1e-3
        rad_plus0 = self.params['rp'].value * 1e-3
        phase = self.params['phase'].value / 360 * 2 * np.pi

        time_wait = (time_tot) - 2. * (time_fringe)
        omega_c =  (nu_c)*2.*np.pi
        omega_rf = (x)*2.*np.pi
        omega_minus = 2.*np.pi*180
        omega_plus = omega_c - omega_minus
        detuning = x - nu_c
        # Coupling constant
        g = conversions/(time_fringe) *np.pi*250. /1000.     #a_rf in units of "conversion"
        g_p2 = np.power(g, 2.0)
        gamma_p2 = np.power((pressure), 2.0)
        detuning_dash = detuning + 1j * 2 * pressure
        # Rabi frequency of the interconversion
        omega_R_dash = (np.sqrt(np.power(detuning_dash, 2.0) + 4.0 * g_p2))
        omega_R_dash_abs_p2 = np.power(abs(omega_R_dash), 2.0)
        d_p2 = 26.4e-3
        omega_z_p2 = (1.602177e-19 * 10) / (d_p2 * 1.660520e-27) / mass
        omega_1_p2 = np.power(omega_c, 2.0) - 2 * omega_z_p2
        
        M_SQRT1_2 = 1.0*10**(-8.5)
        gamma1_tilde = M_SQRT1_2* np.sqrt(np.sqrt(np.power(omega_1_p2 - 4 * gamma_p2, 2.0)+ 16 * gamma_p2 * np.power(omega_c, 2.0))- (omega_1_p2 - 4 * gamma_p2))
        conv_1 = np.cos((.5 * detuning * time_wait) + 1j * ( - pressure * time_wait))
        conv_2 = np.sin((.5 * detuning * time_wait) + 1j * ( - pressure * time_wait))
        ding = detuning_dash / omega_R_dash
        conversion = np.exp(-gamma1_tilde*(2.0*time_fringe+time_wait))*8.5*(4.0 * g_p2 / omega_R_dash_abs_p2)* np.power(abs(conv_1 * np.sin(omega_R_dash * time_fringe)+ ding * conv_2* (np.cos(omega_R_dash * time_fringe) - 1.0)),2.0)
        
        energy_plus = 0.5 * mass * 1.660520e-27 / 1.602177e-19 *conversion* np.power((rad_minus0 * omega_plus), 2.)
        return energy_plus

    def _ramseyasym(self, x, peak=1):
        conversions = self.params['conversions'].value
        time_fringe1 = self.params['Tf1'].value * 1e-3
        time_fringe2 = self.params['Tf2'].value * 1e-3
        time_tot = self.params['Ttot'].value * 1e-3
        nu_c = self.params['f{:.0f}'.format(peak)].value
        mass = self.params['mass'].value
        rad_minus0 = self.params['rm'].value * 1e-3
        rad_plus0 = self.params['rp'].value * 1e-3
        phase = self.params['phase'].value / 360 * 2 * np.pi

        time_wait = time_tot - time_fringe1 - time_fringe2
        omega_c =  nu_c*2.*np.pi
        omega_rf = x*2.*np.pi
        delta = omega_rf - omega_c
        g = conversions /((time_fringe1) + (time_fringe2))*(np.pi)*500./1000.
        omega_R = np.sqrt(np.power(2.*g ,2.0) + np.power(delta, 2.0))
        aa1=np.cos(delta*time_fringe1/2.0)*(np.sin(omega_R*(time_fringe1+time_fringe2)/2.0))
        aa2=delta/omega_R*(np.sin(delta*time_wait/2.0))*(np.cos(omega_R*(time_fringe1+time_fringe2)/2.0)-np.cos(omega_R*(time_fringe1 - time_fringe2)/2.))
        aa3=np.power(2.*g/omega_R*(np.sin(delta*time_wait/2.0))*(np.sin(omega_R*(time_fringe1+time_fringe2)/2.0)), 2.0)
        conversion = np.power(2.*g/omega_R, 2.0) *np.power(aa1 + aa2, 2.0) + aa3
        omega_plus = omega_c #Assumes w_+ ~ w_-
        energy_plus=conversion*0.5*mass*1.660520e-27/1.602177e-19*np.power((rad_minus0*omega_plus),2.)
        return energy_plus

    def konig(self, x):
        mu = self._magneticMoment(x, peak=1)
        tof = self._getTofFaster(mu/7)
        return tof

    def konigbkg(self, x):
        mu1 = self._magneticMoment(x, peak=1)
        tof1 = self._getTofFaster(mu1/7)
        tof2 = self.params['Tofcrap']
        ratio = self.params['ratio'].value / 100
        return tof2*ratio + tof1 * (1 - ratio)

    def doublekonig(self, x):
        mu1 = self._magneticMoment(x, peak=1)
        tof1 = self._getTofFaster(mu1/7)
        mu2 = self._magneticMoment(x, peak=2)
        tof2 = self._getTofFaster(mu2/7)
        ratio = self.params['ratio'].value / 100
        return tof2*ratio + tof1 * (1 - ratio)

    def triplekonig(self, x):
        mu1 = self._magneticMoment(x, peak=1)
        tof1 = self._getTofFaster(mu1/7)
        mu2 = self._magneticMoment(x, peak=2)
        tof2 = self._getTofFaster(mu2/7)
        mu3 = self._magneticMoment(x, peak=3)
        tof3 = self._getTofFaster(mu3/7)
        ratio1 = self.params['ratio'].value / 100
        ratio2 = self.params['ratio2'].value / 100
        return tof3 * ratio2 + tof2*ratio1 + tof1 * (1 - ratio1 - ratio2)

    def ramsey(self, x):
        mu = self._ramsey(x, peak=1)
        tof = self._getTofFaster(mu/7)
        return tof

    def dampedramsey(self, x):
        mu = self._ramseydamp(x, peak=1)
        tof = self._getTofFaster(mu/7)
        return tof

    def asymmetricramsey(self, x):
        mu = self._ramseyasym(x, peak=1)
        tof = self._getTofFaster(mu/7)
        return tof

    def f(self, x):
        x = self.transform(x)
        bkg = self.params['bkg'].value
        bkgratio = self.params['bkgratio'].value / 100
        return self.mapping[self.profile](x) * (1 - bkgratio) + bkg * bkgratio
