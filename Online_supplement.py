"""
Calculation of incremental and total displacements
Equations details in Crispin et al_online supplement_v1

***DISCLAIMER***

**Instructions***
"""

# builtin imports
from dataclasses import dataclass
# other imports
import numpy as np  # tested version: X.X.X
# if script run directly, matplotlib (tested version X.X.X) will also be imported later

def example():
    # if this file is ran manually, this will run the problem in the paper

    figname = "Online_supplement_output.svg"  # where to save the output figure
    
    # calculate EI
    # EI Parameters from Ground Engineering (1984)
    conc_E = 3.1*10**7
    steel_E = 2.1*10**8 
    steel_I_total = 5041866675 + 446688928 # I beam + Steel reinforcement
    conc_I_male = np.pi/4*(590**4) # I of male pile
    nu_wall = 0.2  # Poisson's ratio used to get the plane-strain stiffness
    # EI in kNm^2/m (plane strain conditions)
    EI = (conc_E*conc_I_male + steel_E*steel_I_total) *10**(-12)/(1.95*(1-nu_wall**2))
    # create analysis with required parameters
    disp = Disp(
        "Example analysis",
        cu0 = 40, # Vardanega et al. (2012a) - (by eye fit)
        cu_var = 11, # Vardanega et al. (2012a) - (by eye fit)
        gamma_sat = 20, # Vardanega et al. (2012b)
        b = 0.58, # Vardanega and Bolton (2001a) - mean
        gamma_50 = 0.0070, # Vardanega and Bolton (2001a) - mean
        L = 25+4.6,  # Simpson and Vardanega (2014, pg. 106)
        La = 25+4.6,  # Simpson and Vardanega (2014, pg. 106)
        EI = EI,  # calculated above
        Hm = [5.2, 10.3, 15.1, 19.9, 24.9],  # Depths as calculated below from mAOD levels
        Hp = [0, 4.6, 9.7, 14.5, 19.3],  # 0.6m above last excavation level (Simpson and Vardanega 2014 pg. 106)
        alpha_lamb = 1.2,  # chosen for this model
    )
    disp.run()  # run analysis
    # extract displacement data for plotting
    AOD = 19.5  # level of top of wall
    incr = disp.incremental_disp(AOD=AOD, num_points=100)
    total = disp.total_disp(AOD=AOD, num_points=100)

    import matplotlib.pyplot as plt

    # plot results
    fig, axs = plt.subplots(1, 2, sharex=False, sharey=True, figsize=(7,4), dpi=300, constrained_layout=True)
    # incremental first
    ax = axs[0]
    plt.sca(ax)
    for m in range(1, 1 + disp.num_stages):
        plt.plot(incr[:, m] * 1e3, incr[:, 0], label=f"Stage {m}")
    plt.ylabel("Reduced level (mAOD)")
    plt.xlabel(" ")
    ax.invert_yaxis()
    
    ax = axs[1]
    plt.sca(ax)
    for m in range(1, 1 + disp.num_stages):
        plt.plot(total[:, m] * 1e3, total[:, 0], label=f"Stage {m}")

    plt.ylim(*plt.ylim()[::-1])
    plt.legend(frameon=False)
    plt.figtext(0.5,0.0, "Wall deflection (mm)", ha="center", va="bottom")
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
    cu0: float          # Undrained shear strength at the undisturbed ground surface (kPa)
    cu_var: float       # Variation of the undarined shear strength with depth (kPa) - positive down
    gamma_sat: float    # Saturated unit weight of the soil (kN/m^3)
    b: float            # Soil non-linearity exponent (defined by Vardanega et al 2011)
    gamma_50: float     # Shear strain when 50% of the soil shear strength is mobilised
    # Wall parameters
    L: float            # Wall Length (m)
    La: float           # Wall active length (m)
    EI: float           # Wall Young's Modulus (kPa) multiplied by the second moment of area (m^4)
    # Excavation parameters
    Hm: list[float]  # Excavation depths - measured as 0m at ground surface
    Hp: list[float]  # Prop depths - measured as 0m at ground surface
    # Analysis parameters
    alpha_lamb: float = 1  # Wall fixity conditions
    Mc: float = 2  # Similarity factor
    
    def __post_init__(self, **kwargs):
        """Sets up the mechanism sizes (and related parameters) for each stage"""
        self.num_stages = len(self.Hp)  # Number of excavation stages
        self.deltawmax = []  # list for storing results
        self.deltawmax_cf = []  # list for storing closed from results
        self.hp = [] # Distance between prop level and excavation depth (Figure A9)
        self.lamb = [] # Distance from the prop to the stiff stratum (Figure A9)
        for m in range(0, self.num_stages):
            self.hp.append(self.Hm[m] - self.Hp[m])
            s_m = self.La - self.Hp[m] 
            self.lamb.append(s_m*self.alpha_lamb) if self.Hm[m] > 0 else print("error: Excavation Depth equals 0")

    def run(self, gen_bulging=True, **kwargs):
        """
        Runs the analysis and stores the results in the object
        set gen_bulging to False to turn off the iterative solution
        **kwargs passed to bulging calc
        """
        rot = self.rotation()  # stage 1
        # store result
        self.deltawmax.append(rot)
        self.deltawmax_cf.append(rot)
        for m in range(1, self.num_stages):  # stages 2 onwards
            deltawmax_cf = self.closedform_bulging()
            self.deltawmax_cf.append(deltawmax_cf)  # store result
            if gen_bulging:  # run as normal
                deltawmax = self.bulging(deltawmax_cf)
            else:  # use closed form value
                deltawmax = deltawmax_cf
            self.deltawmax.append(deltawmax)  # store result

    def rotation(self):
        """Calculation of the maximum displacement at the top of the wall during the first stage of excavation"""
        ## Equation A1
        return (
            self.L * self.gamma_50 / 2 * ((self.gamma_sat * self.Hm[0]) * ((3 - 3 * self.Hm[0] / self.L + self.Hm[0] ** 2 / self.L ** 2) /
            (self.cu0 * (6 - 6 * self.Hm[0] / self.L + 3 * self.Hm[0] ** 2 / self.L ** 2) +
            self.cu_var * (2 * self.L - 3 * self.Hm[0] ** 2 / self.L + 2 * self.Hm[0] ** 3 / self.L ** 2)))) ** (1 / self.b)
        )

    def closedform_bulging(self):
        """Closed-form solution for bulging stages (assumes b = 0.5)"""
        m = len(self.deltawmax)  # 0 indexed (1 is Stage 2)
            
        ## Potential Energy Loss
        A = self.gamma_sat * self.lamb[m] * (1 / 2 * self.Hp[m] + 1 / 4 * self.hp[m]) # Eq. A6
        
        ## Internal Elastoplastic Energy
        sum = 0
        for stage_sum in range(1, m):
            sum = sum + self.deltawmax_cf[stage_sum] / self.lamb[stage_sum]
        B1 = self.Mc / (4 * self.gamma_50) * sum # Eq. A12b
        B2 = self.Mc / (4 * self.lamb[m] * self.gamma_50) # Eq. A12c
        B0 = 2*self.Hp[m]/self.lamb[m] + 3*np.pi/8 + 1 # Eq. A7b
        Bvar = self.Hp[m]*(self.Hp[m]/self.lamb[m]+np.pi/4) + np.sqrt(2)*(self.lamb[m]-self.hp[m])*((5*np.pi**2-12)/(16*np.pi**2)) \
            + self.lamb[m]*((4*np.pi**2-16)/(16*np.pi**2)) + self.Hm[m]*(1+np.pi/8) # Eq. A7c
        bm = self.lamb[m]*(B0*self.cu0+Bvar*self.cu_var) # Eq. A13b
        
        ## Elastic Strain Energy
        C1 = np.pi**4*self.EI/self.lamb[m]**3*(1/self.alpha_lamb+1/(4*np.pi)*np.sin(4*np.pi/self.alpha_lamb)) # Eq. A8b
        sum = 0
        for stage_sum in range(1, m):
            sum = sum + self.deltawmax_cf[stage_sum] * self.lamb[m]/self.lamb[stage_sum] * (
                1/(self.lamb[stage_sum]/self.lamb[m]+1)*np.sin(4*np.pi/self.alpha_lamb) + \
                2/(1-(self.lamb[stage_sum]/self.lamb[m])**2)*np.sin(2*np.pi/self.alpha_lamb*(self.lamb[m]/self.lamb[stage_sum]-1))
            )
        C2 = np.pi**3*self.EI/self.lamb[m]**3*sum # Eq. A8c
        
        ## Maximum incremental displacements
        return (
            1/(2*C1**2) * 
            (bm**2*B2 + 2*(A-C2)*C1 - 
            np.sqrt(bm**4*B2**2+4*bm**2*B2*C1*A+4*bm**2*B1*C1**2-4*bm**2*B2*C1*C2))) # Eq. A13a

    def bulging(self, guess: float, tol: float = 0.001, iter_param: int = 1000):
        """
        General solution for bulging stages
        guess is the initial value to use for the deltawmax iteration
        tol (default 0.001) is the tolerance at which the iteration stops
        iter_param (default 1000) controls how much to change the current deltawmax in each iteration
        formula used: new_guess = old_guess + (output - old_guess) / iter parameter
        if the result is not converging, higher values may be required as the solution can be sensitive
        lower values will increase speed
        """
        m = len(self.deltawmax)  # 0 indexed (1 is Stage 2)

        error = 100 # Initialise percentage error between guess and output

        ## Potential Energy Loss
        A = self.gamma_sat*self.lamb[m]*(1/2*self.Hp[m] + 1/4*self.hp[m]) # Eq. A6

        ## Elastic Strain Energy
        C1 = np.pi**4*self.EI/self.lamb[m]**3*(1/self.alpha_lamb+1/(4*np.pi)*np.sin(4*np.pi/self.alpha_lamb)) # Eq. A8b
        sum = 0
        for stage_sum in range(1, m):
            sum = sum + self.deltawmax[stage_sum] * self.lamb[m]/self.lamb[stage_sum] * (
                1/(self.lamb[stage_sum]/self.lamb[m]+1)*np.sin(4*np.pi/self.alpha_lamb) + \
                2/(1-(self.lamb[stage_sum]/self.lamb[m])**2)*np.sin(2*np.pi/self.alpha_lamb*(self.lamb[m]/self.lamb[stage_sum]-1))
            )
        C2 = np.pi ** 3 * self.EI / self.lamb[m] ** 3 * sum  # Eq. A8c
        while error > tol:  # loop until guess is within tolerance
            ## Internal Elastoplastic Energy
            sum = 0
            for stage_sum in range(1, m):
                sum = sum + self.Mc*self.deltawmax[stage_sum]/self.lamb[stage_sum]
            gamma_ave = sum + self.Mc*guess/self.lamb[m] # Eq. A11
            beta_m = 0.5*(gamma_ave/self.gamma_50)**self.b # Eq. A10
            B0 = 2*self.Hp[m]/self.lamb[m] + 3*np.pi/8 + 1 # Eq. A7b
            Bvar = self.Hp[m]*(self.Hp[m]/self.lamb[m]+np.pi/4) + np.sqrt(2)*(self.lamb[m]-self.hp[m])*((5*np.pi**2-12)/(16*np.pi**2)) \
                + self.lamb[m]*((4*np.pi**2-16)/(16*np.pi**2)) + self.Hm[m]*(1+np.pi/8) # Eq. A7c
            B = beta_m*self.lamb[m]*(B0*self.cu0+Bvar*self.cu_var) # Eq. A7a
            output = (A-B-C2)/C1 # Maximum incremental displacements (Eq. A9)
            ## Iteration Process
            error = abs((guess-output)/output*100) # Percentage error between guess and output
            guess = guess + (output-guess)/iter_param # New initial guess of maximum incremental displacements
        return output  # return result

    def soil_mob(self):
        """return the soil mobilisation factor and gamma_ave for each stage"""
        gamma_aves = []
        beta_ms = []
        for m in range(1, len(self.deltawmax) + 1):
            sum = 0
            for stage_sum in range(1, m):
                sum = sum + self.Mc*self.deltawmax[stage_sum]/self.lamb[stage_sum]
            gamma_ave = sum + self.Mc*self.deltawmax[m]/self.lamb[m] # Eq. A11
            beta_m = 0.5*(gamma_ave/self.gamma_50)**self.b # Eq. A10
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
        if AOD is provided, this is converted to reduced level with AOD being teh value at the top of the wall
        """

        incr = np.zeros((num_points, 1 + self.num_stages))
        y = np.linspace(0, self.L, num_points)
        if AOD is None:
            incr[:, 0] = y
        else:
            incr[:, 0] = AOD - y

        # rotation stage
        incr[:, 1] = self.deltawmax[0] * (1 - y / self.La)

        # bulging stages
        for m in range(1, self.num_stages):
            incr[:, m + 1] = (
                (y >= self.Hp[m]) * 
                1 / 2 * (1 - np.cos(2 * np.pi * (y - self.Hp[m]) / self.lamb[m])) * 
                self.deltawmax[m]
            )
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
        total[:, 1:] = np.cumsum(total[:, 1:], axis=1)
        return total

if __name__ == "__main__":
    # script was run
    example()  # run example analysis