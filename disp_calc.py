import numpy as np

"""Calculation of inheritance displacements"""
class disp():
    def __init__(self, name, soil, wall, stage_no, AOD, Hm, Hp, iter_param=400):
        """
        Problem Name [String]
        Soil Parameters [dictionary]:
            "cu0",          # Undrained shear strength at the undisturbed ground surface (kPa)
            "cu_var",       # Variation of the undarined shear strength with depth (kPa) - positive down
            "gamma_sat",    # Saturated unit weight of the soil (kN/m^3)
            "chi",          # Dimensionless parameter within the soil consitutive model
            "b",            # Soil non-linearity exponent (defined by Vardanega et al 2011)
            "gamma_50",     # Shear strain when 50% of the soil shear strength is mobilised
            "k",            # G0=k*cu - used in the elastic solution
        Wall Parameters [dictionary]:
            "L",            # Wall Length (m)
            "C",            # Wall active length (m)
            "alpha",        # Wall fixity conditions
            "EI",           # Wall Young's Modulus (kPa) multiplied by the second moment of area (m^4)
            "nu",           # Poisson's ratio of the wall
        Number of excavation stages [int]
        mAOD of top of wall
        Excavation depths - measured as 0m at ground surface [list]
        Prop depths - measured as 0m at ground surface [list]
        """
        for param, val in soil.items():
            setattr(self, param, val)
        for param, val in wall.items():
            setattr(self, param, val)
        self.name = name
        self.stage_no = stage_no
        self.Hm = Hm
        self.Hp = Hp
        self.AOD = AOD
        self.iter_param = iter_param

        self.hp = [] # Distance between prop level and excavation depth (Figure A9)
        self.lamb = [] # Distance from the prop to the stiff stratum (Figure A9)
        for m in range(0, self.stage_no):
            self.hp.append(self.Hm[m] - self.Hp[m])
            s_m = self.C - self.Hp[m]
            self.lamb.append(s_m*self.alpha) if self.Hm[m] > 0 else print("error: Excavation Depth equals 0")

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
        ## Equation B26
        self.rot = self.gamma_sat * (self.Hm[0]*self.L/(6*self.k)) * \
                            (3*self.L**2-3*self.Hm[0]*self.L+self.Hm[0]**2) / \
                            (self.cu0*(2*self.L**2-2*self.L*self.Hm[0]+self.Hm[0]**2)+\
                            self.cu_var*(2/3*self.L**3-self.Hm[0]**2*self.L+2/3*self.Hm[0]**3))
        
        # print(f'Elastic Rotation = {round(self.rot*1000,2)}mm')

    def rotation(self):
        """Calculation of the maximum displacement at the top of the wall during the first stage of excavation"""
        ## Equation B21
        self.rot = self.L*self.gamma_50/2 * ((self.gamma_sat*self.Hm[0]/(2*self.chi)) * \
                            ((3-3*self.Hm[0]/self.L+self.Hm[0]**2/self.L**2)/ \
                            (self.cu0*(6-6*self.Hm[0]/self.L+3*self.Hm[0]**2/self.L**2)+ \
                            self.cu_var*(2*self.L-3*self.Hm[0]**2/self.L+2*self.Hm[0]**3/self.L**2))))**(1/self.b)
        
        # print(f'Non-linear Rotation = {round(self.rot*1000,2)}mm')

    def closedform_bulging(self):
        self.delta_closedform = [0]
        self.delta_closedform_mm = [0]
        self.beta_m_closedform = [0]
        self.gamma_ave_closedform = [0]
        for m in range(1,self.stage_no):

            ## Equation A26
            a = 1/2*self.Hp[m]/self.lamb[m] + 1/4*self.hp[m]/self.lamb[m]
            ## Equation A25
            A = a*self.gamma_sat*self.lamb[m]**2
            
            ## Equation A59b
            sum = 0
            for stage_sum in range(1, m):
                sum = sum + self.delta_closedform[stage_sum]/self.lamb[stage_sum]
            B1 = 2*self.chi**2/self.gamma_50 * sum
            ## Equation A59c
            B2 = 2*self.chi**2/(self.lamb[m]*self.gamma_50)
            ## Equation A53a
            bhom = 2*self.Hp[m]/self.lamb[m] + np.pi/4 + np.pi/8 + 1
            ## Equation A53b
            binh = self.Hp[m]**2/self.lamb[m] + (np.pi**2-4)/(4*np.pi**2)*self.lamb[m] + \
                        np.pi/4*self.Hp[m] + (np.pi**2-4)/(8*np.pi**2)*np.sqrt(2)*(self.lamb[m]-self.hp[m]) + \
                        np.pi/8*self.Hm[m] + (3*np.pi**2-4)/(16*np.pi**2)*np.sqrt(2)*(self.lamb[m]-self.hp[m]) + \
                        self.Hm[m]
            ## Equation A52 (given B_star=B/beta_m)
            B_star = self.lamb[m]*(bhom*self.cu0+binh*self.cu_var)
            
            ## Equation A6
            C1 = np.pi**4*self.EI/self.lamb[m]**3*(1/self.alpha+1/(4*np.pi)*np.sin(4*np.pi/self.alpha))
            ## Equation A7
            sum = 0
            for stage_sum in range(1, m):
                sum = sum + self.delta_closedform[stage_sum] * self.lamb[m]/self.lamb[stage_sum] * (
                    1/(self.lamb[stage_sum]/self.lamb[m]+1)*np.sin(4*np.pi/self.alpha) + \
                    2/(1-(self.lamb[stage_sum]/self.lamb[m])**2)*np.sin(2*np.pi/self.alpha*(self.lamb[m]/self.lamb[stage_sum]-1))
                )
            C2 = np.pi**3*self.EI/self.lamb[m]**3*sum
            
            
            ## Equation A62 - maximum incremental displacements
            delta = 1/(2*C1**2) * (B_star**2*B2 + 2*(A-C2)*C1 - np.sqrt(B_star**4*B2**2+4*B_star**2*B2*C1*A+4*B_star**2*B1*C1**2-4*B_star**2*B2*C1*C2))
            self.delta_closedform.append(delta)
            self.delta_closedform_mm.append(round(delta*1000,2))

            ## Equation A59a - mobilisation factor
            beta_m = np.sqrt(B1+B2*self.delta_closedform[m])
            self.beta_m_closedform.append(beta_m)

            ##Equation A57 - average shear strain in the mechanism
            sum = 0
            for stage_sum in range(1, m):
                sum = sum + 2*self.delta_closedform[stage_sum]/self.lamb[stage_sum]
            gamma_ave = sum
            self.gamma_ave_closedform.append(gamma_ave)

        # print("Closed-form dispalcements (mm) = ", self.delta_closedform_mm)

    def bulging(self):
        self.delta = [0]
        self.delta_mm = [0]
        self.beta_m = [0]
        self.gamma_ave = [0]
        for m in range(1,self.stage_no):
            input = self.delta_closedform[m] # Initial guess of displacements
            error = 100 # Percentage error between input and output

            ## Equation A26
            a = 1/2*self.Hp[m]/self.lamb[m] + 1/4*self.hp[m]/self.lamb[m]
            ## Equation A25
            A = a*self.gamma_sat*self.lamb[m]**2

            ## Equation A6
            C1 = np.pi**4*self.EI/self.lamb[m]**3*(1/self.alpha+1/(4*np.pi)*np.sin(4*np.pi/self.alpha))
            ## Equation A7
            sum = 0
            for stage_sum in range(1, m):
                sum = sum + self.delta[stage_sum] * self.lamb[m]/self.lamb[stage_sum] * (
                    1/(self.lamb[stage_sum]/self.lamb[m]+1)*np.sin(4*np.pi/self.alpha) + \
                    2/(1-(self.lamb[stage_sum]/self.lamb[m])**2)*np.sin(2*np.pi/self.alpha*(self.lamb[m]/self.lamb[stage_sum]-1))
                )
            C2 = np.pi**3*self.EI/self.lamb[m]**3*sum

            while error > 0.01:
                ## Equation A57
                sum = 0
                for stage_sum in range(1,m):
                    sum = sum + 2*self.delta[stage_sum]/self.lamb[stage_sum]
                gamma_ave = sum + 2*input/self.lamb[m]
                ## Equation 56
                beta_m = self.chi*(gamma_ave/self.gamma_50)**self.b
                ## Equation A53a
                bhom = 2*self.Hp[m]/self.lamb[m] + np.pi/4 + np.pi/8 + 1
                ## Equation A53b
                binh = self.Hp[m]**2/self.lamb[m] + (np.pi**2-4)/(4*np.pi**2)*self.lamb[m] + \
                            np.pi/4*self.Hp[m] + (np.pi**2-4)/(8*np.pi**2)*np.sqrt(2)*(self.lamb[m]-self.hp[m]) + \
                            np.pi/8*self.Hm[m] + (3*np.pi**2-4)/(16*np.pi**2)*np.sqrt(2)*(self.lamb[m]-self.hp[m]) + \
                            self.Hm[m]
                ## Equation A52
                B = beta_m*self.lamb[m]*(bhom*self.cu0+binh*self.cu_var)
                
                ## Equation 55
                output = (A-B-C2)/C1

                ## Iteration Process
                error = abs((input-output)/output*100)
                input = input + (output-input)/self.iter_param

            ## Maximum incremental displacements
            self.delta.append(output)
            self.delta_mm.append(round(output*1000,2))

            ## Mobilisation factor and average shear strain within the mechanism
            self.gamma_ave.append(gamma_ave)
            self.beta_m.append(beta_m)


        # print("General solution displacements (mm) = ", self.delta_mm)

    def pre_plot(self, num_points):

        self.y_points = np.linspace(0,self.L,num_points)

        self.rotation_vals()
        self.incremental_vals()
        self.total_vals()
    
    def rotation_vals(self):
        self.x_points_rotation = []
        for y in self.y_points:
            grad = -self.L/self.rot
            x = (y-self.L)/grad
            self.x_points_rotation.append(x)

    def incremental_vals(self):
        self.x_points_inc = {}
        for m in range(1,self.stage_no):
            inc = []
            for y in self.y_points:
                if y < self.Hp[m]:
                    x = 0
                else:
                    x = 1/2*(1-np.cos(2*np.pi*(y-self.Hp[m])/self.lamb[m]))*self.delta[m]
                inc.append(x)
            self.x_points_inc[m] = inc

    def total_vals(self):
        self.x_points_total = {}
        for m in range(1,self.stage_no):
            tot = self.x_points_rotation
            for stage_sum in range(1,m+1):
                tot = np.add(tot, self.x_points_inc[stage_sum])
            self.x_points_total[m] = tot