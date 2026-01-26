"""
Online Supplement: MSD applied to the construction of the British Library basement: a multi-stage excavation in London Clay.

Authors: J.J Crispin, A.H. Bateman, E. Voyagaki, A. Campbell, G. Mylonakis, M.D. Bolton and P.J. Vardanega.

Suggested citation:
Crispin, J.J., Bateman, A.H., Voyagaki, E., Campbell, A., Mylonakis, G., Bolton, M.D. & Vardanega, P.J. (2023). 
MSD applied to the construction of the British Library basement: a multistage excavation in London Clay. 
Canadian Geotechnical Journal, https://doi.org/10.1139/cgj-2023-0238

This python code calculates maximum incremental wall displacements for each stage using the general solution (or closed-form solution),
and plots the resultant incremental and total wall displacements along the wall length.
Equation numbers and notation are provided in the accompnaying online supplement.

This code is tested on Python version 3.9.12.

*** DISCLAIMER ***
This code is provided by the authors to show the derivation of the MSD predictions presented in the main text.
It is not intended to be used or relied upon (in whole or part) for any other purpose, and no warranty is provided or implied.
While every effort has been made, the authors cannot guarentee that this code is error free.

*** NOTES ***
1. Input parameters for this example are provided in the example function.
2. Initialise with problem parameters and run Disp.run() to run the analysis.
3. The general solution is run automatically (with iteration which can be controlled with iter_param if not converging).
4. The closed form solution can be chosen instead by running Disp.run(gen_bulging=False).
5. Resulting deltawmax values are stored in Disp.deltawmax.
6. Disp.soil_mob() returns the soil mobilisation factor and gamma_ave for each stage.
7. Disp.incremental_disp() returns the incremental displacement profile with depth for each stage.
8. Disp.total_disp() returns the total displacement profile with depth for each stage.
9. The incremental and total displacements for each stage are automatically plotted in the saved figure.

*** LICENSE ***
Copyright (c) 2023 J.J Crispin, A.H. Bateman, E. Voyagaki, A. Campbell, G. Mylonakis, M.D. Bolton and P.J. Vardanega

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

__version__ = "1.0.0"

# builtin imports
from dataclasses import dataclass
# other imports
import numpy as np  # tested version: 1.21.5
# if script run directly, matplotlib (tested version 3.5.2) will also be imported later

def example():
    """
    Function defines the input parameters and runs the analysis for the example problem.
    A plot of incremental and total displacements for each stage is produced.

    If this file is ran manually, this will run the problem in the paper.
    """

    figname = "Online_supplement_output.svg"  # where to save the output figure
    
    ## RUN ANALYSIS WITH INPUT PARAMETERS
    # calculate wall stiffness EI (EI Parameters from Ground Engineering (1984))
    conc_E = 3.1*10**7
    steel_E = 2.1*10**8 
    steel_I_total = 5041866675 + 446688928 # I beam + Steel reinforcement
    conc_I_male = np.pi/4*(590**4) # I of male pile
    nu_wall = 0.2  # Poisson's ratio used to get the plane-strain stiffness
    # EI in kNm^2/m (plane strain conditions)
    EI = (conc_E*conc_I_male + steel_E*steel_I_total) *10**(-12)/(1.95*(1-nu_wall**2)) # kNm^2/m (plane-strain conditions)
    # create analysis with required parameters (parameter definitions provided in object)
    disp = Disp(
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
    Class to run the analysis
    Initialise with problem parameters and run "Disp.run()" to run the analysis
    Resulting deltawmax values are stored in Disp.deltawmax
    Disp.soil_mob() returns the soil mobilisation factor and gamma_ave for each stage
    Disp.incremental_disp() returns the incremental displacement profile with depth for each stage
    Disp.total_disp() returns the total displacement profile with depth for each stage
    """
    name: str  # Name of analysis
    # Soil parameters
    su0: float          # Undrained shear strength at the undisturbed ground surface (kPa)
    su_var: float       # Variation of the undarined shear strength with depth (kPa)
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
    alpha_lamb: float = 1  # Wall fixity conditions
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

    def closedform_bulging(self):
        """Closed-form solution for incremental maximum displacements during bulging stages (assumes b = 0.5)"""
        m = len(self.deltawmax)  # 0 indexed (1 is Stage 2)
            
        ## Potential Energy Loss
        A = self.gamma_sat * self.lamb[m] * (1 / 2 * self.Hp[m] + 1 / 4 * self.hp[m]) # Eq. A6
        
        ## Internal Elastoplastic Energy
        sum = 0
        for stage_sum in range(1, m):
            sum = sum + self.deltawmax_cf[stage_sum] / self.lamb[stage_sum]
        chi_1 = self.Mc / (4 * self.gamma_50) * sum # Eq. A12b
        chi_2 = self.Mc / (4 * self.lamb[m] * self.gamma_50) # Eq. A12c
        b_0 = 2 * self.Hp[m] / self.lamb[m] + 3 * np.pi / 8 + 1 # Eq. A7b
        b_var = (self.Hp[m] * (self.Hp[m] / self.lamb[m] + np.pi / 4) + np.sqrt(2) * (self.lamb[m] - self.hp[m]) * 
            ((5 * np.pi ** 2 - 12) / (16 * np.pi ** 2)) + self.lamb[m]*((4 * np.pi ** 2 - 16) / (16 * np.pi ** 2)) 
            + self.Hm[m] * ( 1 + np.pi / 8)) # Eq. A7c
        B_max = self.lamb[m] * (b_0 * self.su0 + b_var * self.su_var) # Eq. A13b
        
        ## Elastic Strain Energy
        C1 = np.pi ** 4 * self.EI / self.lamb[m] ** 3 * (
            (1 / self.alpha_lamb + 1 / (4 * np.pi) * np.sin(4 * np.pi / self.alpha_lamb))) # Eq. A8b
        sum = 0
        for stage_sum in range(1, m):
            sum = sum + (self.deltawmax_cf[stage_sum] / (self.lamb[stage_sum] ** 3 * (1 + self.lamb[m] / self.lamb[stage_sum])) * (
                2 / (self.lamb[m] / self.lamb[stage_sum] - 1) * np.sin(2 * np.pi / self.alpha_lamb * (self.lamb[m] / self.lamb[stage_sum] - 1)) + 
                self.lamb[stage_sum] / self.lamb[m] * np.sin(4 * np.pi / self.alpha_lamb))
                )
        C2 = np.pi ** 3 * self.EI * sum # Eq. A8c
        
        ## Maximum incremental displacements
        return (
            1 / (2 * C1 ** 2) * 
            (B_max ** 2 * chi_2 + 2 * C1 * (A - C2) - B_max *
            np.sqrt(B_max ** 2 * chi_2 ** 2 + 4 * chi_2 * C1 * A + 4 * chi_1 * C1 ** 2 - 4 * chi_2 * C1 * C2))) # Eq. A13a

    def bulging(self, guess: float, tol: float = 0.001, iter_param: int = 1000):
        """
        General solution for incremental maximum displacements during bulging stages
        guess is the initial value to use for the deltawmax iteration (suggested to set to closed form solution)
        tol (default 0.001) is the tolerance at which the iteration stops
        iter_param (default 1000) controls how much to change the current deltawmax in each iteration
        formula used: new_guess = old_guess + (output - old_guess) / iter parameter
        if the result is not converging, higher values may be required as the solution can be sensitive
        lower values will increase speed
        """
        m = len(self.deltawmax)  # 0 indexed (1 is Stage 2)

        error = 100 # Initialise percentage error between guess and output

        ## Potential Energy Loss
        A = self.gamma_sat * self.lamb[m] * (1 / 2 * self.Hp[m] + 1 / 4 * self.hp[m]) # Eq. A6

        ## Elastic Strain Energy
        C1 = np.pi ** 4 * self.EI / self.lamb[m] ** 3 * (
            (1 / self.alpha_lamb + 1 / (4 * np.pi) * np.sin(4 * np.pi / self.alpha_lamb))) # Eq. A8b
        sum = 0
        for stage_sum in range(1, m):
            sum = sum + (self.deltawmax[stage_sum] / (self.lamb[stage_sum] ** 3 * (1 + self.lamb[m] / self.lamb[stage_sum])) * (
                2 / (self.lamb[m] / self.lamb[stage_sum] - 1) * np.sin(2 * np.pi / self.alpha_lamb * (self.lamb[m] / self.lamb[stage_sum] - 1)) + 
                self.lamb[stage_sum] / self.lamb[m] * np.sin(4 * np.pi / self.alpha_lamb))
                )
        C2 = np.pi ** 3 * self.EI * sum # Eq. A8c

        while error > tol:  # loop until guess is within error tolerance
            ## Internal Elastoplastic Energy
            sum = 0
            for stage_sum in range(1, m):
                sum = sum + self.Mc * self.deltawmax[stage_sum] / self.lamb[stage_sum]
            gamma_ave = sum + self.Mc * guess / self.lamb[m] # Eq. A10b
            beta_m = 1 / 2 * (gamma_ave / self.gamma_50) ** self.b # Eq. A10a
            b_0 = 2 * self.Hp[m] / self.lamb[m] + 3 * np.pi / 8 + 1 # Eq. A7b
            b_var = (self.Hp[m] * (self.Hp[m] / self.lamb[m] + np.pi / 4) + np.sqrt(2) * (self.lamb[m] - self.hp[m]) * 
                ((5 * np.pi ** 2 - 12) / (16 * np.pi ** 2)) + self.lamb[m]*((4 * np.pi ** 2 - 16) / (16 * np.pi ** 2)) 
                + self.Hm[m] * ( 1 + np.pi / 8)) # Eq. A7c
            B = beta_m * self.lamb[m] * (b_0 * self.su0 + b_var * self.su_var) # Eq. A7a
            output = (A - B - C2) / C1 # Maximum incremental displacements (Eq. A9)

            ## Iteration Process
            error = abs((guess - output) / output * 100) # Percentage error between guess and output
            guess = guess + (output - guess) / iter_param # New initial guess of maximum incremental displacements
        
        return output  # return result

    def soil_mob(self):
        """return the soil mobilisation factor and gamma_ave for each stage"""
        gamma_aves = [2 * self.deltawmax[0] / self.L]  # gamma_ave for rotation stage (2*theta)
        beta_ms = [1 / 2 * (gamma_aves[0] / self.gamma_50) ** self.b] # Eq. 10a
        for m in range(1, len(self.deltawmax)): # Calculation for bulging stages
            sum = 0
            for stage_sum in range(1, m):
                sum = sum + self.Mc * self.deltawmax[stage_sum] / self.lamb[stage_sum]
            gamma_ave = sum + self.Mc * self.deltawmax[m] / self.lamb[m] # Eq. A10b
            beta_m = 1 / 2 * (gamma_ave / self.gamma_50) ** self.b # Eq. A10a
            gamma_aves.append(gamma_ave)
            beta_ms.append(beta_m)
        return beta_ms, gamma_aves

    def incremental_disp(self,
        AOD: float = None,  # Reduced level mAOD of top of wall, depth used if equals None (default)
        num_points=100,  # number of points to split into
    ):
        """
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
        returns an np.array of the total displacements at each depth for each stage
        the first column is the depth (split into num_points)
        if AOD is provided, this is converted to reduced level with AOD being teh value at the top of the wall
        """

        total = self.incremental_disp(AOD, num_points)
        total[:, 1:] = np.cumsum(total[:, 1:], axis=1) # Eq. A11
        return total

if __name__ == "__main__":
    # script was run
    example()  # run example analysis