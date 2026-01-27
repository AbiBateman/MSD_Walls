"""
Code Authors: Crispin, J.J., Bateman, A.H., Struzik, Z., Campbell, A., Voyagaki, E., Mylonakis, G., Vardanega, P.J.
Released: January 2026
If you are using this code, please cite the relevant publications below.

Version "v2.0.0" of this code has been used in the following publication:
Hua, Y., Struzik, Z., Bateman, A.H., Crispin, J.J., Huang, H., Zhang, D., Mylonakis, G.E. & Vardanega, P.J. (2026).
Assessing the effectiveness of the mobilizable strength design method for prediction of retaining wall movements using a database.
Submitted.

Version "v1.0.0" of this code has been used in the following publication:
Crispin, J.J., Bateman, A.H., Voyagaki, E., Campbell, A., Mylonakis, G., Bolton, M.D. & Vardanega, P.J. (2024). 
MSD applied to the construction of the British Library basement: a multistage excavation in London Clay. 
Canadian Geotechnical Journal, 61(3):596-603.  https://doi.org/10.1139/cgj-2023-0238

This code is available at the following repository: https://github.com/AbiBateman/MSD_Walls

This code is tested on Python version 3.12.3.

*** DISCLAIMER ***
This code is provided by the authors to show the derivation of the MSD wall predictions.
It is not intended to be used or relied upon (in whole or part) for any other purpose, and no warranty is provided or implied.
While every effort has been made, the authors cannot guarentee that this code is error free.

*** NOTES ***
This python code calculates maximum incremental wall displacements and location for each stage using the general solution (or closed-form solution),
and plots the resultant incremental and total wall displacements along the wall length.

1. Input parameters for this example are provided in the example function.
2. Initialise with problem parameters and run Disp.run() to run the analysis.
3. The general solution is run automatically (with iteration which can be controlled with iter_param if not converging).
4. The closed form solution can be chosen instead by running Disp.run(gen_bulging=False).
5. Resulting maximum displacement values, deltawmax, are stored in Disp.deltawmax.
6. Disp.soil_mob() returns the soil mobilisation factor and gamma_ave for each stage.
7. Disp.incremental_disp() returns the incremental displacement profile with depth for each stage.
8. Disp.total_disp() returns the total displacement profile with depth for each stage.
9. The incremental and total displacements for each stage are automatically plotted in the saved figure.

*** LICENSE ***
Copyright (c) 2026 Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

__version__ = "2.0.0"

# builtin imports
from dataclasses import dataclass
from abc import ABC, abstractmethod
# other imports
import numpy as np  # tested version: 1.26.4
# if script run directly, matplotlib (tested version 3.9.2) will also be imported later
from scipy.optimize import root_scalar, bisect  ## tested version: 1.13.1

def example(numerical=False):
    """
    Function defines the input parameters and runs the analysis for an example
    problem, the British Library excavation analysed in Crispin et al. (2024).
    A plot of incremental and total displacements for each stage is produced.
    Analytical or numerical solutions can be employed (through the "numerical" argument)
    """

    figname = "MSD_walls_output.svg"  # where to save the output figure
    
    ## RUN ANALYSIS WITH INPUT PARAMETERS
    
    # calculate wall stiffness EI (EI Parameters from Ground Engineering (1984))
    conc_E = 3.1*10**7
    steel_E = 2.1*10**8 
    steel_I_total = 5041866675 + 446688928 # I beam + Steel reinforcement
    conc_I_male = np.pi/4*(590**4) # I of male pile
    nu_wall = 0.2  # Poisson's ratio used to get the plane-strain stiffness
    # EI in kNm^2/m (plane strain conditions)
    EI = (conc_E*conc_I_male + steel_E*steel_I_total) *10**(-12)/(1.95*(1-nu_wall**2)) # kNm^2/m (plane-strain conditions)

    # select analysis type (analytical or numerical)
    Disp_class = Disp_Num if numerical else Disp

    # create analysis with required parameters (parameter definitions provided in object)
    disp = Disp_class(
        "Example analysis", # Title of problem
        su0 = 40, # Vardanega et al. (2012a) - (by eye fit)
        su_var = 11, # Vardanega et al. (2012a) - (by eye fit)
        gamma_sat = 20, # Vardanega et al. (2012b)
        b = 0.58, # Vardanega and Bolton (2001a) - mean
        gamma_50 = 0.0070, # Vardanega and Bolton (2001a) - mean
        L = 25+4.6,  # Simpson and Vardanega (2014, pg. 106)
        EI = EI,  # calculated above
        Hm = [5.2, 10.3, 15.1, 19.9, 24.9],  # Depths as calculated below from mAOD levels
        Hp = [0, 4.6, 9.7, 14.5, 19.3],  # 0.6m above last excavation level (Simpson and Vardanega 2014 pg. 106)
        alpha_lamb = 1.2,  # chosen for this model
    )
    disp.run()  # run analysis

    # extract displacement data for plotting
    AOD = 19.5  # reduced level of top of wall
    print("Maximum incremental displacements (closed form) (m)")
    print(np.round(disp.deltawmax_cf,4)) ## prints maximum incremental displacements (closed form)
    print("Maximum incremental displacements (general form) (m)")
    print(np.round(disp.deltawmax,4)) ## prints maximum incremental displacements (general form)
    incr = disp.incremental_disp(AOD=AOD, num_points=100)  # calculate incremental displacements along length of wall
    total = disp.total_disp(AOD=AOD, num_points=100)  # calculate total displacements along length of wall

    ## PLOT RESULTS
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, sharex=False, sharey=True, figsize=(7,4), dpi=300, constrained_layout=True)
    # incremental displacements
    ax = axs[0]
    plt.sca(ax)
    for m in range(1, 1 + disp.num_stages):
        plt.plot(incr[:, m] * 1e3, incr[:, 0], label=f"Stage {m}")
    plt.ylabel("Reduced level (mAOD)")
    plt.xlabel(" ")
    plt.annotate("Incremental Displacements", [0.98, 0.02], xycoords="axes fraction", ha="right", va="bottom")
    ax.invert_yaxis()

    x_all = total[:, 1:] * 1e3  # All displacements (all stages)
    y_all = total[:, 0]         # All depths (same for all stages)

    # Find position of absolute maximum displacement
    max_pos = np.argmax(x_all)
    # Get the corresponding depth (y-value)
    y_at_max = y_all[max_pos // (disp.num_stages)]
    # Get the actual maximum displacement value
    max_disp = x_all.max()
    print(max_disp, "Maximum displacement (mm)")
    print(AOD-y_at_max, "Depth of displacement (m)")
    
    
    ax = axs[1]
    # total displacements
    plt.sca(ax)
    for m in range(1, 1 + disp.num_stages):
        plt.plot(total[:, m] * 1e3, total[:, 0], label=f"Stage {m}")

    plt.ylim(*plt.ylim()[::-1])
    plt.legend(frameon=False)
    plt.figtext(0.5,0.0, "Wall deflection (mm)", ha="center", va="bottom")
    plt.annotate("Total Displacements", [0.98, 0.02], xycoords="axes fraction", ha="right", va="bottom")
    plt.savefig(figname)

    return disp  # return resulting Disp object

@dataclass
class Disp():
    """
    Class to run the analysis using an analytical solution to the DeltaP and DeltaW integrals
    Initialise with problem parameters and run "Disp.run()" to run the analysis
    Resulting deltawmax values are stored in Disp.deltawmax
    Disp.soil_mob() returns the soil mobilisation factor and gamma_ave for each stage
    Disp.incremental_disp() returns the incremental displacement profile with depth for each stage
    Disp.total_disp() returns the total displacement profile with depth for each stage
    """
    name: str  # Name of analysis
    # Soil parameters
    su0: float          # Undrained shear strength at the undisturbed ground surface (kPa)
    su_var: float       # Variation of the undarined shear strength with depth (kPa/m)
    gamma_sat: float    # Saturated unit weight of the soil (kN/m^3)
    b: float            # Soil non-linearity exponent
    gamma_50: float     # Shear strain when 50% of the soil shear strength is mobilised
    # Wall parameters
    L: float            # Wall Length (m)
    EI: float           # Wall Young's Modulus (kPa) multiplied by the second moment of area (m^4) per meter length (plane strain)
    # Excavation parameters
    Hm: list[float]  # Excavation depths - measured as 0m at ground surface (shown in Figure A3)
    Hp: list[float]  # Prop depths - measured as 0m at ground surface (shown in Figure A3)
    # Analysis parameters
    alpha_lamb: float = 1.14  # Wall fixity conditions, see Hua et al. (2026) for default of 1.14
    Mc: float = 2  # Similarity factor
    
    def __post_init__(self, **kwargs):
        """Sets up the mechanism sizes (and related parameters) for each stage"""
        self.num_stages = len(self.Hp)  # Number of excavation stages
        self.deltawmax = []  # list for storing maximum incremental displacement results
        self.deltawmax_cf = []  # list for storing closed form incremental displacement results
        self.hp = [] # Distance between prop level and excavation depth (Figure A3)
        self.lamb = [] # Wave length of the deformation mechanism (Figure A3)
        for m in range(0, self.num_stages):
            self.hp.append(self.Hm[m] - self.Hp[m]) # Distance between the excavated depth and the last installed prop (Figure A3)
            s_m = self.L - self.Hp[m] # Distance between the last installed prop and the base of the wall (Figure A3)
            self.lamb.append(s_m*self.alpha_lamb) # Wavelength of the deformation mechanism (Figure A3)

    def run(self, gen_bulging=True, **kwargs):
        """
        Runs the analysis and stores the results in the object
        set gen_bulging to False to turn off the iterative solution (gives closed-form bulging solutions)
        **kwargs passed to bulging calc
        """
        rot = self.rotation()  # stage 1
        # store result
        self.deltawmax.append(rot)
        self.deltawmax_cf.append(rot)
        for m in range(1, self.num_stages):  # stages 2 onwards
            deltawmax_cf = self.closedform_bulging()  # obtain closed form maximum incremental displacement value
            self.deltawmax_cf.append(deltawmax_cf)  # store result
            if gen_bulging:  # run as normal
                deltawmax = self.bulging(deltawmax_cf)  # obtain general solution maximum incremental displacement value
            else:  # use closed form value
                deltawmax = deltawmax_cf
            self.deltawmax.append(deltawmax)  # store result

    def rotation(self):
        """Calculation of the maximum displacement at the top of the wall during the first stage of excavation"""
        ## Equation A1b
        return (
            self.L * self.gamma_50 / 2 * ((self.gamma_sat * self.Hm[0]) * ((3 - 3 * self.Hm[0] / self.L + self.Hm[0] ** 2 / self.L ** 2) /
            (3 * self.su0 * (2 - 2 * self.Hm[0] / self.L + self.Hm[0] ** 2 / self.L ** 2) +
            self.su_var * self.L * (2 - 3 * self.Hm[0] ** 2 / self.L ** 2 + 2 * self.Hm[0] ** 3 / self.L ** 3)))) ** (1 / self.b)
        )

    def closedform_bulging(self, *, m=None):
        """Closed-form solution for incremental maximum displacements during bulging stages (assumes b = 0.5)"""
        if m is None:  # calculate for next stage
            m = len(self.deltawmax)  # 0 indexed (1 is Stage 2)
            
        ## Calculate energy terms
        A = self.calculate_A()  ## Eq. A6
        C1, C2 = self.calculate_Cs()  ## Eq. A9
        Bmax = self.calculate_Bmax()  ## Eq. A7
        
        ## calculate chi_1 and chi_2
        sum = 0
        for stage_sum in range(1, m):
            sum = sum + self.deltawmax_cf[stage_sum] / self.lamb[stage_sum]
        chi_1 = self.Mc / (4 * self.gamma_50) * sum # Eq. A13b
        chi_2 = self.Mc / (4 * self.lamb[m] * self.gamma_50) # Eq. A13c

        ## Maximum incremental displacements
        return (
            1 / (2 * C1 ** 2) * 
            (Bmax ** 2 * chi_2 + 2 * C1 * (A - C2) - Bmax *
            np.sqrt(Bmax ** 2 * chi_2 ** 2 + 4 * chi_2 * C1 * A + 4 * chi_1 * C1 ** 2 - 4 * chi_2 * C1 * C2))
            ) # Eq. A14

    def bulging(self, guess: float, *, tol: float = 0.001, iter_param: int = 1000, m=None):
        """
        General solution for incremental maximum displacements during bulging stages
        guess is the initial value to use for the deltawmax iteration (suggested to set to closed form solution)
        tol (default 0.001) is the tolerance at which the iteration stops
        iter_param (default 1000) controls how much to change the current deltawmax in each iteration
        formula used: new_guess = old_guess + (output - old_guess) / iter parameter
        if the result is not converging, higher values may be required as the solution can be sensitive
        lower values will increase speed
        """
        if m is None:  # calculate for next stage
            m = len(self.deltawmax)  # 0 indexed (1 is Stage 2)

        error = 100 # Initialise percentage error between guess and output

        ## Calculate energy terms
        A = self.calculate_A()  ## Eq. A6
        C1, C2 = self.calculate_Cs()  ## Eq. A9
        Bmax = self.calculate_Bmax()  ## Eq. A7

        while error > tol:  # loop until guess is within error tolerance
            ## Internal Elastoplastic Energy
            sum = 0
            for stage_sum in range(1, m):
                sum = sum + self.Mc * self.deltawmax[stage_sum] / self.lamb[stage_sum]
            gamma_ave = sum + self.Mc * guess / self.lamb[m] # Eq. A11b
            beta_m = 1 / 2 * (gamma_ave / self.gamma_50) ** self.b # Eq. A11a
            
            B = beta_m * Bmax # Eq. A7a
            output = (A - B - C2) / C1 # Maximum incremental displacements (Eq. A10)

            ## Iteration Process
            error = abs((guess - output) / output * 100) # Percentage error between guess and output
            guess = guess + (output - guess) / iter_param # New initial guess of maximum incremental displacements
        
        return output  # return result

    def calculate_A(self, *, m=None):
        """
        Calculate A for bulging stage m. This is the potential energy loss, DeltaPm, normalised by deltaw_max
        a is calculated first for each zone, such that DeltaPm=a*lambda**2*gamma_sat*deltaw_max.
        """
        if m is None:  # calculate for next stage
            m = len(self.deltawmax)  # 0 indexed (1 is Stage 2)

        ## Parameter Normalisation
        Hp_bar = self.Hp[m] / self.lamb[m]
        hp_bar = self.hp[m] / self.lamb[m]

        ## Calculation of a (Table A1)
        a = 1 / 4 * (1 + 2 * Hp_bar - (1 - hp_bar) ** 2 + 1 / np.pi**2 * np.sin(np.pi * hp_bar) ** 2)  ## Table A1

        return a * self.gamma_sat * self.lamb[m] ** 2  ## Eq. A6

    def calculate_Cs(self, *, m=None):
        """
        Calculate C1 and C2 for bulging stage m. These is the elastic strain energy in the wall, DeltaUm, normalised by deltaw_max such that
        DeltaPm=C1*deltaw_max^2+C2*deltaw_max^2
        """
        if m is None:  # calculate for next stage
            m = len(self.deltawmax)  # 0 indexed (1 is Stage 2)
        
        ## Elastic Strain Energy
        C1 = np.pi ** 4 * self.EI / self.lamb[m] ** 3 * (
            (1 / self.alpha_lamb + 1 / (4 * np.pi) * np.sin(4 * np.pi / self.alpha_lamb))) # Eq. A9b
        sum = 0
        for stage_sum in range(1, m):
            sum = sum + (self.deltawmax[stage_sum] / (self.lamb[stage_sum] ** 3 * (1 + self.lamb[m] / self.lamb[stage_sum])) * (
                2 / (self.lamb[m] / self.lamb[stage_sum] - 1) * np.sin(2 * np.pi / self.alpha_lamb * (self.lamb[m] / self.lamb[stage_sum] - 1)) + 
                self.lamb[stage_sum] / self.lamb[m] * np.sin(4 * np.pi / self.alpha_lamb))
                )
        C2 = np.pi ** 3 * self.EI * sum # Eq. A9c

        return C1, C2


    def calculate_Bmax(self, *, m=None):
        """
        Calculate Bmax for bulging stage m. This is the internal Plastic work, DeltaWm, normalised by deltaw_max and beta, the soil strength mobilisation factor
        b_0 and b_var are calculated first for each zone, such that DeltaWm=beta_m*lambda*(b_0*s0+b_var*su_var*y)*deltaw_max.
        """
        if m is None:  # calculate for next stage
            m = len(self.deltawmax)  # 0 indexed (1 is Stage 2)

        # Parameter normalisation
        Hp_bar = self.Hp[m] / self.lamb[m]
        H_bar = self.Hm[m] / self.lamb[m]
        hp_bar = self.hp[m] / self.lamb[m]

        # helper functions for finding where sign of gamma changes to solve B integrals
        def gamma(r , lamb , r0=0):
            """
            shear strain at distance r for specific mechanism (defined by lambda and hp)
            multiplied by r to avoid the singularity at zero when finding roots - this makes finding r2 easier with a negative first bracket
            for zone CDE, set r0=0 as the centre is point D
            for zone EFH, set r0=hp as the centre is point F
            """
            return r * (np.pi / lamb * np.sin(2*np.pi * (r + r0) / lamb) - 1 / 2 / r * (1-np.cos(2*np.pi * (r + r0) / lamb)))  ## Eq. A8a multiplied by r

        def deriv_gamma(r , lamb , hp):
            """derivative of shear strain (dg/dr) at distance r for specific mechanism (defined by lambda and hp)"""
            return 1 / (2 * r**2) * (
                1 - (1 - 4 * np.pi**2 * r**2 / lamb**2) * np.cos(2 * np.pi * (r + hp) / lamb)
                - 2 * np.pi * r / lamb * np.sin(2 * np.pi * (r + hp) / lamb)
            )  ## Eq. A8b

        # Zone ABCD (Table A2 & A3)
        b_0_abcd = 2 * Hp_bar
        b_var_abcd = Hp_bar**2

        # Zone CDE  (Table A2 & A3)
        # find r1 using gamma equation (ignoring hp)
        r1 = root_scalar(gamma, method="bisect", args=(self.lamb[m], 0), bracket=(0.01* self.lamb[m] , (0.4 * self.lamb[m]))).root
        r1_bar = r1 / self.lamb[m]
        b_0_cde = 1/2 * (np.sin(2 * np.pi * r1_bar) - 2 * np.pi * r1_bar * np.cos(np.pi * r1_bar)**2 + np.pi)
        b_var_cde = 1 / (4 * np.pi**2) * (
            6 * np.pi * r1_bar * np.sin(2 * np.pi * r1_bar) - 3 * (1 - np.cos(2 * np.pi * r1_bar))
            + np.pi**2 * (3 - 4 * r1_bar**2 * np.cos(2 * np.pi * r1_bar) - 2 * r1_bar**2)
            + 2 * np.pi**2 * Hp_bar * (np.pi - np.pi * r1_bar * (1 + np.cos(2 * np.pi * r1_bar)) + np.sin(2 * np.pi * r1_bar))
        )

        # Zone EFH (Table A2 & A3)
        if (
            self.hp[m] > 0.2 * self.lamb[m] or  # remove majority of cases to make root finding easier
            gamma(rp := root_scalar(deriv_gamma, method="bisect", args=(self.lamb[m], self.hp[m]), bracket=(0.001,0.4 * self.lamb[m])).root, self.lamb[m], self.hp[m]) < 0
        ):  # check if sign change in gamma occurs, store rp for later use
            # no sign change
            b_0_efh = 1 / 8 * (np.sin(2 * np.pi * hp_bar) - 2 * np.pi * (hp_bar - 1))
            b_var_efh = 1 / (16 * np.pi**2) * (
                3 * np.sqrt(2) * (np.cos(2 * np.pi * hp_bar) - 1) + 4 * np.pi**3 * H_bar * (1 - hp_bar)
                + 2 * np.pi**2 * (H_bar * np.sin(2 * np.pi * hp_bar) + 3 * np.sqrt(2) * (hp_bar - 1)**2)
            )
        else:
            # sign change, find two roots
            r2 = root_scalar(gamma, method="bisect", args=(self.lamb[m], self.hp[m]), bracket=(-0.001 * self.lamb[m], rp)).root  # first non-zero root
            r3 = root_scalar(gamma, method="bisect", args=(self.lamb[m], self.hp[m]), bracket=(rp ,(0.4 * self.lamb[m]))).root  # second non-zero root
            r2_bar = r2 / self.lamb[m]
            r3_bar = r3 / self.lamb[m]
            # calculate b_0 and b_var
            b_0_efh = 1 / 8 * (
                + 2 * np.pi * (1 - hp_bar) + np.sin(2 * np.pi * hp_bar)
                + 2 * np.sin(2 * np.pi*(hp_bar + r3_bar)) - 2 * np.sin(2 * np.pi * (hp_bar + r2_bar))
                + 2 * np.pi * r2_bar * (np.cos(2 * np.pi * (hp_bar + r2_bar)) + 1)
                - 2 * np.pi * r3_bar * (np.cos(2 * np.pi * (hp_bar + r3_bar)) + 1)
            )
            cr2 = np.cos(2 * np.pi * (hp_bar + r2_bar))
            cr3 = np.cos(2 * np.pi * (hp_bar + r3_bar))
            sr2 = np.sin(2 * np.pi * (hp_bar + r2_bar))
            sr3 = np.sin(2 * np.pi * (hp_bar + r3_bar))
            b_var_efh = 1/(16*np.pi**2) * (
                2 * np.sqrt(2) * np.pi**2 * (2 * r2_bar**2 - 2 * r3_bar**2 + 3 * (1 - hp_bar)**2)
                + 4 * np.pi**3 * H_bar * (1 - hp_bar + r2_bar - r3_bar)
                - 3 * np.sqrt(2)
                + 3 * np.sqrt(2) * (np.cos(2 * np.pi * hp_bar) - 2 * (cr2 - cr3))
                + 2 * np.pi**2 * H_bar * (np.sin(2 * np.pi * hp_bar) - 2 * (sr2 - sr3))
                + 4 * np.pi**3 * H_bar * (r2_bar * cr2 - r3_bar * cr3)
                - 12 * np.sqrt(2) * np.pi * (r2_bar * sr2 - r3_bar* sr3)
                + 8 * np.sqrt(2) * np.pi**2 * (r2_bar**2 * cr2 - r3_bar**2 * cr3)
            )

        # Zone FHJ (Table A2 & A3)
        b_0_fhj = 1 / (4 * np.pi) * (4 * np.pi - np.sin(2 * np.pi * hp_bar) - 6 * np.pi * hp_bar)
        b_var_fhj = 1 / (16 * np.pi**2) * (
            + np.pi**2 * (3 * np.sqrt(2) + 16 * H_bar - 24 * hp_bar * H_bar + 6 * np.sqrt(2) * hp_bar**2 - 8 * np.sqrt(2) * hp_bar)
            - 4 * np.pi * H_bar * np.sin(2 * np.pi * hp_bar)
            - 2 * np.sqrt(2) * (np.cos(np.pi * hp_bar)**2 + 1)
        )

        # Total values across the mechanism
        b_0 = b_0_abcd + b_0_cde + b_0_efh + b_0_fhj
        b_var = b_var_abcd + b_var_cde + b_var_efh + b_var_fhj

        # convert into Bmax
        Bmax = self.lamb[m] * (b_0 * self.su0 + b_var * self.lamb[m] * self.su_var) # Eq. A7

        return Bmax

    def soil_mob(self):
        """
        convenience function for extracting results/plotting
        return the soil mobilisation factor and gamma_ave for each stage
        """
        gamma_aves = [2 * self.deltawmax[0] / self.L]  # gamma_ave for rotation stage (2*theta)
        beta_ms = [1 / 2 * (gamma_aves[0] / self.gamma_50) ** self.b] # Eq. 11a
        for m in range(1, len(self.deltawmax)): # Calculation for bulging stages
            sum = 0
            for stage_sum in range(1, m):
                sum = sum + self.Mc * self.deltawmax[stage_sum] / self.lamb[stage_sum]
            gamma_ave = sum + self.Mc * self.deltawmax[m] / self.lamb[m] # Eq. A11b
            beta_m = 1 / 2 * (gamma_ave / self.gamma_50) ** self.b # Eq. A11a
            gamma_aves.append(gamma_ave)
            beta_ms.append(beta_m)
        return beta_ms, gamma_aves

    def incremental_disp(self,
        AOD: float = None,  # Reduced level mAOD of top of wall, depth used if equals None (default)
        num_points=100,  # number of points to split into
    ):
        """
        convenience function for extracting results/plotting
        returns an np.array of the incremental displacements at each depth for each stage
        the first column is the depth (split into num_points)
        if AOD is provided, this is converted to reduced level with AOD being the value at the top of the wall
        """

        incr = np.zeros((num_points, 1 + self.num_stages))
        y = np.linspace(0, self.L, num_points)  # Depths along length of wall
        if AOD is None:
            incr[:, 0] = y  # Depths along length of wall
        else:
            incr[:, 0] = AOD - y  # Reduced levels

        # rotation stage
        incr[:, 1] = self.deltawmax[0] * (1 - y / self.L) # Eq. A1a

        # bulging stages
        for m in range(1, self.num_stages):
            incr[:, m + 1] = (
                (y >= self.Hp[m]) * 1 / 2 * (1 - np.cos(2 * np.pi * (y - self.Hp[m]) / self.lamb[m])) * self.deltawmax[m]
            ) # Eq. A2
        return incr

    def total_disp(self,
        AOD: float = None,  # Reduced level mAOD of top of wall, depth used if equals None (default)
        num_points=100,  # number of points to split into
    ):
        """
        convenience function for extracting results/plotting
        returns an np.array of the total displacements at each depth for each stage
        the first column is the depth (split into num_points)
        if AOD is provided, this is converted to reduced level with AOD being teh value at the top of the wall
        """

        total = self.incremental_disp(AOD, num_points)
        total[:, 1:] = np.cumsum(total[:, 1:], axis=1) # Eq. A12
        return total

@dataclass
class Disp_Num(Disp):
    """
    Class to run the analysis using a numerical solution to the DeltaP and DeltaW integrals
    NOTE - rotation stage still uses the analytical solution for linearly varying su and constant gamma_sat
    Initialise with problem parameters and run "Disp.run()" to run the analysis
    Resulting deltawmax values are stored in Disp.deltawmax
    Disp.soil_mob() returns the soil mobilisation factor and gamma_ave for each stage
    Disp.incremental_disp() returns the incremental displacement profile with depth for each stage
    Disp.total_disp() returns the total displacement profile with depth for each stage
    """

    # Soil parameters, either define as func, or give su0 and su_var (for su) and gamma_sat directly
    su_func: callable = None        # variation of su with depth, y
    gamma_sat_func: callable = None # variation of gamma_sat with depth, y
    # su0, su_var and gamma_sat are still needed for the rotation stage, so must still be defined.
    # These will default to None in future versions when solved.
    ### su0: float = None               # Undrained shear strength at the undisturbed ground surface (kPa)
    ### su_var: float = None            # Variation of the undarined shear strength with depth (kPa/m)
    ### gamma_sat: float = None         # Saturated unit weight of the soil (kN/m^3)

    # Analysis parameters
    element_size: float = 0.01      # maximum element dimension for numerical integration as fraction of lambda

    def __post_init__(self, **kwargs):
        '''ensure parameter functions are defined correctly'''

        # ensure su_func is defined correctly
        if self.su_func is None:  # use su0 and su_var to define su, otherwise these parameters are ignored
            if self.su0 is None or self.su_var is None:
                raise ValueError("If su_func is not explicitly defined, su0 and su_var must be instead")
            else:
                self.su_func = lambda y: self.su0 + self.su_var * y
        # ensure gamma_sat_func is defined correctly
        if self.gamma_sat_func is None:  # use su0 and su_var to define su, otherwise these parameters are ignored
            if self.gamma_sat is None:
                raise ValueError("If gamma_sat_func is not explicitly defined gamma_sat must be instead")
            else:
                self.gamma_sat_func = lambda y: self.gamma_sat

        # run inherited __post_init__ code
        super().__post_init__(**kwargs)

    def calculate_A(self, *, m=None):
        """
        Calculate A for bulging stage m. This is the potential energy loss, DeltaPm, normalised by deltaw_max
        It is calculated using numerical integration for each zone, then summed together
        """
        if m is None:  # calculate for next stage
            m = len(self.deltawmax)  # 0 indexed (1 is Stage 2)
        
        # get required parameters for this stage
        params = dict(
            element_size=self.element_size,
            Hm=self.Hm[m],
            Hp=self.Hp[m],
            lamb=self.lamb[m],
            su_func=self.su_func,
            gamma_sat_func=self.gamma_sat_func,
        )

        # set up numerical solvers for each zone
        solvers = [
            ABCD(**params),
            CDE(**params),
            EFH(**params),
            FHJ(**params),
        ]
        # loop through and sum A from each solver
        A = np.sum([sol.A() for sol in solvers])

        return A
    
    def calculate_Bmax(self, *, m=None):
        """
        Calculate A for bulging stage m. This is the potential energy loss, DeltaPm, normalised by deltaw_max
        It is calculated using numerical integration for each zone, then summed together
        """
        if m is None:  # calculate for next stage
            m = len(self.deltawmax)  # 0 indexed (1 is Stage 2)
        
        # get required parameters for this stage
        params = dict(
            element_size=self.element_size,
            Hm=self.Hm[m],
            Hp=self.Hp[m],
            lamb=self.lamb[m],
            su_func=self.su_func,
            gamma_sat_func=self.gamma_sat_func,
        )

        # set up numerical solvers for each zone
        solvers = [
            ABCD(**params),
            CDE(**params),
            EFH(**params),
            FHJ(**params),
        ]
        # loop through and sum Bmax from each solver
        Bmax = np.sum([sol.Bmax() for sol in solvers])

        return Bmax

@dataclass
class MSD_numerical(ABC):
    """
    Class to solve integrals required for A and Bmax calculations numerically
    Splits 2D domain into grid of certain elementsize
    Local grid (xi and eta) used for defining grid and ease of defining displacements and calculating strain
    Assumed to be a cartesian grid, function should be provided to convert if not
    Function provided in order to convert this into a global grid (x and y)
    """
    element_size: float  # maximum element size (normalised by lambda)
    lamb: float  # lambda for specfic calculation
    Hm: float  # Hm for specific calculation
    Hp: float  # Hp for specific calculation
    su_func: callable
    gamma_sat_func: callable

    def __post_init__(self, **kwargs):

        # unnormalise element size
        self.element_size *= self.lamb

        # create local grid and get local coordinates
        self.xis, self.etas = self.create_local_grid()
        self.xs, self.ys = self.convert_to_global_grid(self.xis, self.etas)

        # vectorize functions of su and gamma_sat
        self.su_func = np.vectorize(self.su_func)
        self.gamma_sat_func = np.vectorize(self.gamma_sat_func)
        # calculate functions of grid
        self.su = self.su_func(self.ys)
        self.gamma_sat = self.gamma_sat_func(self.ys)
    
    @staticmethod
    def disp_func(xis, etas, lamb):
        """Assumed (normalised) displacement function (defined in 2 dimensions)"""
        us = np.zeros_like(xis)
        vs = 0.5 * (1 - np.cos(2 * np.pi * xis / lamb))
        return us, vs
    
    @property
    def hp(self):
        """Value of hp calculated from other dimensions"""
        return self.Hm - self.Hp
    
    @property
    @abstractmethod
    def maxxi(self):
        """Local grid (xi) assumed to go from 0 to this maximum value"""
        ...

    @property
    @abstractmethod
    def maxeta(self):
        """Local grid (eta) assumed to go from 0 to this maximum value"""
        ...
    
    @property
    def nxi(self):
        """Number of points in local grid (xi coordinate)"""
        return -int(-self.maxxi // self.element_size)  # ceil
    
    @property
    def neta(self):
        """Number of points in local grid (eta coordinate)"""
        return -int(-self.maxeta // self.element_size)  # ceil
    
    @property
    def dxi(self):
        """Spacing of local grid (xi coordinate)"""
        return self.maxxi / self.nxi
    
    @property
    def deta(self):
        """Spacing of local grid (eta coordinate)"""
        return self.maxeta / self.neta
    
    def create_local_grid(self):
        """Creates a grid for each zone in local coordinates relevant to the zone"""

        initxi = self.dxi / 2
        endxi = self.maxxi - initxi

        initeta= self.deta / 2
        endeta= self.maxeta - initeta

        xi_list = np.linspace(initxi, endxi,  self.nxi)
        eta_list = np.linspace(initeta, endeta, self.neta)
        xis, etas = np.meshgrid(xi_list, eta_list)

        return xis, etas
    
    @property
    def dA(self):
        """grid defining the area of each element in the zone"""
        return self.dxi * self.deta * np.ones_like(self.xis)
    
    def displacements(self):
        """Returns grid of u and v displacements in local coordinate system"""
        return self.disp_func(self.xis, self.etas, self.lamb)
    
    def convert_to_cartesian(self, xis, etas):
        """Converts local grid to cartesian coordinates with x defined as right and y defined as downwards"""
        return xis, etas

    def convert_to_global_grid(self, xis, etas):
        """
        Converts local grid into global coordinate system, a cartesian grid with 
        x and y defined as distance right and downwards of point A, respectively
        Used to get material properties on local grid
        """
        return xis, etas

    def local_strains(self):
        """
        Calculate the strains due to the displacements in the local grid
        Calculation assumes cartesian coordinates
        """
        # get displacements and numerical derivatives
        us, vs = self.displacements()
        du_deta, du_dxi = np.gradient(us, self.deta, self.dxi)
        dv_deta, dv_dxi = np.gradient(vs, self.deta, self.dxi)

        # calculate strains from standard cyclindrical definitions
        epsuu = du_dxi
        epsvv = dv_deta
        gamuv = du_deta + dv_dxi

        return epsuu, epsvv, gamuv

    def principle_strains(self):
        """Calculate the principle strains due to the displacements"""
        # get local strains
        epsuu, epsvv, gamuv = self.local_strains()

        # get extreme normal strains from Mohr's circle formulas
        eps1 = (epsuu + epsvv) / 2 + np.sqrt(((epsuu - epsvv) / 2) ** 2 + (gamuv / 2) ** 2)
        eps3 = (epsuu + epsvv) / 2 - np.sqrt(((epsuu - epsvv) / 2) ** 2 + (gamuv / 2) ** 2)

        # gamma is diameter of Mohr's circle of strain
        gam = eps1 - eps3

        return eps1, eps3, gam

    def Area(self):
        """Calculate the area of the zone (useful for checking)"""
        # simply sum the area of each element
        return np.sum(self.dA)

    def A(self):
        """Calculate A=DeltaP/deltaw_max by summing the contributions of each element"""
        us, vs = self.convert_to_cartesian(*self.displacements())  # convert displacements to cartesian coordinates
        return np.sum(vs * self.dA * self.gamma_sat)  # numerically integrate

    def Bmax(self):
        """Calculate Bmax=DeltaW/(beta*deltaw_max) by summing the contributions of each element"""
        _, _, gam = self.principle_strains()  # calculate gamma for each element
        return np.sum(self.su * np.abs(gam) * self.dA)  # numerically integrate


class MSD_numerical_cylindrical(MSD_numerical, ABC):
    """
    Class to solve integrals required for A and Bmax calculations numerically
    Splits 2D domain into grid of certain elementsize
    Local grid (xi and eta) used for defining grid and ease of defining displacements and calculating strain
    Assumed to be a cylindircal grid, function should be provided to convert to cartesian
    Function provided in order to convert this into a global grid (x and y)
    """

    @property
    def neta(self):  # eta is an angle so element_size needs to be normalised by lambda
        """Number of points in local grid (eta coordinate)"""
        return -int(-self.maxeta * self.lamb // self.element_size)  # ceil
    
    @property
    def dA(self):  # r dr dtheta
        """grid defining the area of each element in the zone"""
        return self.xis * self.dxi * self.deta
    
    def local_strains(self):
        """
        Calculate the strains due to the displacements in the local grid
        Calculation assumes cylindrical coordinates
        """
        # get displacements and numerical derivatives
        us, vs = self.displacements()
        du_deta, du_dxi = np.gradient(us, self.deta, self.dxi)
        dv_deta, dv_dxi = np.gradient(vs, self.deta, self.dxi)

        # calculate strains from standard cyclindrical definitions
        r = self.xis
        eps_rr = du_dxi
        eps_phiphi = us / r + (1 / r) * dv_deta
        gam_rphi = dv_dxi - vs / r + (1 / r) * du_deta

        return eps_rr, eps_phiphi, gam_rphi


class ABCD(MSD_numerical):
    """
    Class to solve integrals required for A and Bmax calculations in zone ABCD numerically
    Splits 2D domain into grid of certain elementsize
    """

    @property
    def maxxi(self):
        """Local grid (xi) assumed to go from 0 to this maximum value"""
        return self.lamb
    
    @property
    def maxeta(self):
        """Local grid (eta) assumed to go from 0 to this maximum value"""
        return self.Hp

    
class CDE(MSD_numerical_cylindrical):
    """
    Class to solve integrals required for A and Bmax calculations in zone CDE numerically
    Splits 2D domain into grid of certain elementsize
    """

    @property
    def maxxi(self):
        """Local grid (xi) assumed to go from 0 to this maximum value"""
        return self.lamb
    
    @property
    def maxeta(self):
        """Local grid (eta) assumed to go from 0 to this maximum value"""
        return np.pi / 2
    
    def convert_to_cartesian(self, us, vs):
        """Converts local grid to cartesian coordinates with x defined as right and y defined as downwards"""
        return -us * np.cos(self.etas) + vs * np.sin(self.etas), us * np.sin(self.etas) + vs * np.cos(self.etas)

    def convert_to_global_grid(self, xis, etas):
        """
        Converts local grid into global coordinate system, a cartesian grid with 
        x and y defined as distance right and downwards of point A, respectively
        Used to get material properties on local grid
        """
        # local coordinates cyclindrical centred at D
        return self.lamb - xis * np.cos(etas), self.Hp + xis * np.sin(etas)
    
class EFH(MSD_numerical_cylindrical):
    """
    Class to solve integrals required for A and Bmax calculations in zone EFH numerically
    Splits 2D domain into grid of certain elementsize
    """

    @property
    def maxxi(self):
        """Local grid (xi) assumed to go from 0 to this maximum value"""
        return self.lamb - self.hp
    
    @property
    def maxeta(self):
        """Local grid (eta) assumed to go from 0 to this maximum value"""
        return np.pi / 4
    
    def displacements(self):
        """Returns grid of u and v displacements in local coordinate system"""
        # xi coordinate increased by hp as rotation actually about prop not top of zone
        # i.e. function starts at D but coordinate system centred at F
        return self.disp_func(self.xis+self.hp, self.etas, self.lamb)
    
    def convert_to_cartesian(self, us, vs):
        """Converts local grid to cartesian coordinates with x defined as right and y defined as downwards"""
        # return -us * np.cos(self.etas + np.pi/2) + vs * np.sin(self.etas + np.pi/2), us * np.sin(self.etas + np.pi/2) + vs * np.cos(self.etas + np.pi/2)
        return us * np.sin(self.etas) + vs * np.cos(self.etas), us * np.cos(self.etas) - vs * np.sin(self.etas)
    
    def convert_to_global_grid(self, xis, etas):
        """
        Converts local grid into global coordinate system, a cartesian grid with 
        x and y defined as distance right and downwards of point A, respectively
        Used to get material properties on local grid
        """
        # local coordinates cyclindrical centred at F
        return self.lamb + xis * np.sin(etas), self.Hm + xis * np.cos(etas)
    
class FHJ(MSD_numerical):
    """
    Class to solve integrals required for A and Bmax calculations in zone FHJ numerically
    Splits 2D domain into grid of certain elementsize
    """
    # displacements function unchanged as coordinate system centred at H

    @property
    def maxxi(self):
        """Local grid (xi) assumed to go from 0 to this maximum value"""
        return self.lamb - self.hp
    
    @property
    def maxeta(self):
        """Local grid (eta) assumed to go from 0 to this maximum value"""
        return self.lamb - self.hp
    
    @property
    def dA(self):
        """grid defining the area of each element in the zone"""
        # triangular zone, so need to remove half the grid
        dA = np.ones_like(self.xis) # assing all values 1
        for i in range(len(dA)):
            dA[i,-i-1:] = 0  # assign zero to bottom right half of grid
            dA[i,-i-1] = 0.5  # assign half to values
        dA *= self.dxi * self.deta  # multiply values by dxi and deta to get dA
        return dA

    def convert_to_cartesian(self, us, vs):
        """Converts local grid to cartesian coordinates with x defined as right and y defined as downwards"""
        # xi goes up and to the right, eta goes up and to the left
        return - us / np.sqrt(2) + vs / np.sqrt(2), - us / np.sqrt(2) - vs / np.sqrt(2)

    def convert_to_global_grid(self, xis, etas):
        """
        Converts local grid into global coordinate system, a cartesian grid with 
        x and y defined as distance right and downwards of point A, respectively
        Used to get material properties on local grid
        """
        # xi goes up and to the right, eta goes up and to the left
        # local coordinates centred at H
        return (
            self.lamb + (self.lamb - self.hp) / np.sqrt(2) - xis / np.sqrt(2) + etas  / np.sqrt(2),
            self.Hm + (self.lamb - self.hp) / np.sqrt(2) - xis  / np.sqrt(2) - etas  / np.sqrt(2),
        )

if __name__ == "__main__":
    # script was run
    example()  # run example analysis