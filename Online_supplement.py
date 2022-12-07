import numpy as np

"""Calculation of incremental displacements - Equations details in Crispin et al_online supplement_v1"""
class disp():
    def __init__(self, name, soil, wall, stage_no, AOD, Hm, Hp, iter_param=400):
        """
        Problem Name [String]
        Soil Parameters [dictionary]:
            "cu0",          # Undrained shear strength at the undisturbed ground surface (kPa)
            "cu_var",       # Variation of the undarined shear strength with depth (kPa) - positive down
            "gamma_sat",    # Saturated unit weight of the soil (kN/m^3)
            "b",            # Soil non-linearity exponent (defined by Vardanega et al 2011)
            "gamma_50",     # Shear strain when 50% of the soil shear strength is mobilised
            "k",            # G0=k*cu - used in the elastic solution
        Wall Parameters [dictionary]:
            "L",            # Wall Length (m)
            "C",            # Wall active length (m)
            "alpha_lamb",        # Wall fixity conditions
            "EI",           # Wall Young's Modulus (kPa) multiplied by the second moment of area (m^4)
            "nu",           # Poisson's ratio of the wall
            "Mc",           # Similarity factor
        Number of excavation stages [int]
        mAOD of top of wall
        Excavation depths - measured as 0m at ground surface [list]
        Prop depths - measured as 0m at ground surface [list]
        Iteration parameter
        """
        for param, val in soil.items(): # Soil parameters
            setattr(self, param, val)
        for param, val in wall.items(): # Wall parameters
            setattr(self, param, val)
        self.name = name # Problem Name
        self.stage_no = stage_no # Number of stages of excavation
        self.Hm = Hm # Depths of excavation
        self.Hp = Hp # Depths of props
        self.AOD = AOD # Reduced level (mAOD) of top of wall
        self.iter_param = iter_param # Iteration paramter

        self.hp = [] # Distance between prop level and excavation depth (Figure A9)
        self.lamb = [] # Distance from the prop to the stiff stratum (Figure A9)
        for m in range(0, self.stage_no):
            self.hp.append(self.Hm[m] - self.Hp[m]) # Distance between previous prop and current excavation depht
            s_m = self.C - self.Hp[m] # Distance from last installed prop to the base of the wall
            self.lamb.append(s_m*self.alpha_lamb) if self.Hm[m] > 0 else print("error: Excavation Depth equals 0") # Wave length of displacement mechanism (Eq. A3)

    def run(self, elastic_rotation=False, gen_bulging=True):
        """Select which methods to run"""
        if elastic_rotation:
            self.elastic_rotation()
        else:
            self.rotation()
        self.closedform_bulging()
        if gen_bulging:
            self.bulging()
        else:
            self.delta = self.delta_closedform
            self.delta_mm = self.delta_closedform_mm
            self.beta_m = self.beta_m_closedform
            self.gamma_ave = self.gamma_ave_closedform
       

    def elastic_rotation(self):
        """Calculation of the maximum displacement at the top of the wall during the first stage of excavation assuming elastic conditions"""
        ## Equation B26 of original Technical Appendix
        self.rot = self.gamma_sat * (self.Hm[0]*self.L/(6*self.k)) * \
                            (3*self.L**2-3*self.Hm[0]*self.L+self.Hm[0]**2) / \
                            (self.cu0*(2*self.L**2-2*self.L*self.Hm[0]+self.Hm[0]**2)+\
                            self.cu_var*(2/3*self.L**3-self.Hm[0]**2*self.L+2/3*self.Hm[0]**3))

    def rotation(self):
        """Calculation of the maximum displacement at the top of the wall during the first stage of excavation"""
        ## Equation A1
        self.rot = self.L*self.gamma_50/2 * ((self.gamma_sat*self.Hm[0]) * ((3-3*self.Hm[0]/self.L+self.Hm[0]**2/self.L**2)/ \
                            (self.cu0*(6-6*self.Hm[0]/self.L+3*self.Hm[0]**2/self.L**2)+ \
                            self.cu_var*(2*self.L-3*self.Hm[0]**2/self.L+2*self.Hm[0]**3/self.L**2))))**(1/self.b)

    def closedform_bulging(self):
        """Closed-form solution for bulging stages (assumes b = 0.5)"""
        self.delta_closedform = [0]
        self.delta_closedform_mm = [0]
        self.beta_m_closedform = [0]
        self.gamma_ave_closedform = [0]
        for m in range(1,self.stage_no):
            
            ## Potential Energy Loss
            A = self.gamma_sat*self.lamb[m]*(1/2*self.Hp[m] + 1/4*self.hp[m]) # Eq. A6
            
            ## Internal Elastoplastic Energy
            sum = 0
            for stage_sum in range(1, m):
                sum = sum + self.delta_closedform[stage_sum]/self.lamb[stage_sum]
            B1 = self.Mc/(4*self.gamma_50) * sum # Eq. A12b
            B2 = self.Mc/(4*self.lamb[m]*self.gamma_50) # Eq. A12c
            B0 = 2*self.Hp[m]/self.lamb[m] + 3*np.pi/8 + 1 # Eq. A7b
            Bvar = self.Hp[m]*(self.Hp[m]/self.lamb[m]+np.pi/4) + np.sqrt(2)*(self.lamb[m]-self.hp[m])*((5*np.pi**2-12)/(16*np.pi**2)) \
                + self.lamb[m]*((4*np.pi**2-16)/(16*np.pi**2)) + self.Hm[m]*(1+np.pi/8) # Eq. A7c
            bm = self.lamb[m]*(B0*self.cu0+Bvar*self.cu_var) # Eq. A13b
            
            ## Elastic Strain Energy
            C1 = np.pi**4*self.EI/self.lamb[m]**3*(1/self.alpha_lamb+1/(4*np.pi)*np.sin(4*np.pi/self.alpha_lamb)) # Eq. A8b
            sum = 0
            for stage_sum in range(1, m):
                sum = sum + self.delta_closedform[stage_sum] * self.lamb[m]/self.lamb[stage_sum] * (
                    1/(self.lamb[stage_sum]/self.lamb[m]+1)*np.sin(4*np.pi/self.alpha_lamb) + \
                    2/(1-(self.lamb[stage_sum]/self.lamb[m])**2)*np.sin(2*np.pi/self.alpha_lamb*(self.lamb[m]/self.lamb[stage_sum]-1))
                )
            C2 = np.pi**3*self.EI/self.lamb[m]**3*sum # Eq. A8c
            
            ## Maximum incremental displacements
            delta = 1/(2*C1**2) * (bm**2*B2 + 2*(A-C2)*C1 - np.sqrt(bm**4*B2**2+4*bm**2*B2*C1*A+4*bm**2*B1*C1**2-4*bm**2*B2*C1*C2)) # Eq. A13a
            self.delta_closedform.append(delta)
            self.delta_closedform_mm.append(round(delta*1000,2))

            ## Mobilisation Factor
            beta_m = np.sqrt(B1+B2*self.delta_closedform[m]) # Eq. A12a
            self.beta_m_closedform.append(beta_m)

            ## Average shear strain in the mechanism
            sum = 0
            for stage_sum in range(1, m+1):
                sum = sum + self.Mc*self.delta_closedform[stage_sum]/self.lamb[stage_sum]
            gamma_ave = sum # Eq. A11
            self.gamma_ave_closedform.append(gamma_ave)

    def bulging(self):
        """General solution for bulging stages"""
        self.delta = [0]
        self.delta_mm = [0]
        self.beta_m = [0]
        self.gamma_ave = [0]
        for m in range(1,self.stage_no):
            input = self.delta_closedform[m] # Initial guess of maximum incremental displacements
            error = 100 # Percentage error between input and output

            ## Potential Energy Loss
            A = self.gamma_sat*self.lamb[m]*(1/2*self.Hp[m] + 1/4*self.hp[m]) # Eq. A6
            
            ## Elastic Strain Energy
            C1 = np.pi**4*self.EI/self.lamb[m]**3*(1/self.alpha_lamb+1/(4*np.pi)*np.sin(4*np.pi/self.alpha_lamb)) # Eq. A8b
            sum = 0
            for stage_sum in range(1, m):
                sum = sum + self.delta[stage_sum] * self.lamb[m]/self.lamb[stage_sum] * (
                    1/(self.lamb[stage_sum]/self.lamb[m]+1)*np.sin(4*np.pi/self.alpha_lamb) + \
                    2/(1-(self.lamb[stage_sum]/self.lamb[m])**2)*np.sin(2*np.pi/self.alpha_lamb*(self.lamb[m]/self.lamb[stage_sum]-1))
                )
            C2 = np.pi**3*self.EI/self.lamb[m]**3*sum # Eq. A8c
            
            while error > 0.001:
                ## Internal Elastoplastic Energy
                sum = 0
                for stage_sum in range(1,m):
                    sum = sum + self.Mc*self.delta[stage_sum]/self.lamb[stage_sum]
                gamma_ave = sum + self.Mc*input/self.lamb[m] # Eq. A11
                beta_m = 0.5*(gamma_ave/self.gamma_50)**self.b # Eq. A10
                B0 = 2*self.Hp[m]/self.lamb[m] + 3*np.pi/8 + 1 # Eq. A7b
                Bvar = self.Hp[m]*(self.Hp[m]/self.lamb[m]+np.pi/4) + np.sqrt(2)*(self.lamb[m]-self.hp[m])*((5*np.pi**2-12)/(16*np.pi**2)) \
                    + self.lamb[m]*((4*np.pi**2-16)/(16*np.pi**2)) + self.Hm[m]*(1+np.pi/8) # Eq. A7c
                B = beta_m*self.lamb[m]*(B0*self.cu0+Bvar*self.cu_var) # Eq. A7a
                
                ## Maximum incremental displacements
                output = (A-B-C2)/C1 # Eq. A9

                ## Iteration Process
                error = abs((input-output)/output*100) # Percentage error between input and output
                input = input + (output-input)/self.iter_param # New initial guess of maximum incremental displacements

            ## Maximum incremental displacements
            self.delta.append(output)
            self.delta_mm.append(round(output*1000,2))

            ## Mobilisation factor and average shear strain within the mechanism
            self.gamma_ave.append(gamma_ave)
            self.beta_m.append(beta_m)


    def pre_plot(self, num_points):
        """x and y values along length of wall for all displacement stages"""
        self.y_points = np.linspace(0,self.L,num_points) # Values along length of wall

        self.x_points_inc = {}
        self.x_points_total = {}

        for m in range(0,self.stage_no):
            self.x_points_inc[m] = []
            for y in self.y_points:
                ## Rotation Stage
                if m == 0:
                    x = (y-self.L)/(-self.L/self.rot)
                    self.x_points_inc[0].append(x)
                
                ## Incremental bulging stages
                else:
                    if y < self.Hp[m]:
                        xb = 0
                    else:
                        xb = 1/2*(1-np.cos(2*np.pi*(y-self.Hp[m])/self.lamb[m]))*self.delta[m]
                    self.x_points_inc[m].append(xb)

            ## Total bulging stages
            tot = self.x_points_inc[0]
            if m != 0:
                for stage_sum in range(1,m+1):
                    tot = np.add(tot, self.x_points_inc[stage_sum])
            self.x_points_total[m] = tot