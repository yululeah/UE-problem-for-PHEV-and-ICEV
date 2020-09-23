from graph import TrafficNetwork, Graph
from excel import ExcelProcessor
import numpy as np
np.set_printoptions(threshold = 1e2) 

class TrafficFlowModel:
    ''' TRAFFIC FLOW ASSIGN MODEL
        Inside the Frank-Wolfe algorithm is given, one can use
        the method `solve` to compute the numerical solution of
        User Equilibrium problem, and the input could be introduced
        into the model both by initialization and by using the
        excel interface.
    '''
    def __init__(self, graph= None, origins= [], destinations= [], 
    demands= None, link_free_time= None, link_capacity= None):

        self.__network = TrafficNetwork(graph= graph, O= origins, D= destinations)

        # Initialization of parameters 初始化
        self.__link_free_time = np.array(link_free_time)
        self.__link_capacity = np.array(link_capacity)
        #self.__demand = np.array(demands)
        
        # Alpha and beta (used in performance function)
        #self._alpha = 0.15
        #self._beta = 4 

        # Convergent criterion 精度
        self._conv_accuracy = 1e-5

        # Boolean varible: If true print the detail while iterations
        #布尔变量： 在迭代中如果正确则打印
        self.__detail = False

        # Boolean varible: If true the model is solved properly
         #布尔变量： 如果被解决则正确
        self.__solved = False

        # Some variables for contemporarily storing the
        # computation result 存储一下时下的变量
        self.__final_link_flow = None
        self.__iterations_times = None

        # Class for excel interface
        self.__ep = ExcelProcessor()
        
    def create_template(self):


        self.__ep.create_template()

    def read_data(self):
        ''' Read the data from created tempalte, whose
            name is `input.xls`. Notice that you must 
            fill in all your data properly before invoking
            this method. Since the compatibility in our
            case is not very good, using this way to
            introduce data into model is not very recommended
        '''
        link_length, link_free_time, link_capacity = self.__ep.read_basic_params()
         #Read the values of variables link_free_time and link_capacity from the file `input.xls`
        origins, destinations, demands = self.__ep.read_demands()
        links = self.__ep.read_links()
        limits = self.__ep.read_limits()


        self.length_limits=len(limits)
        self.__insert_links_in_order(links) 
        #Insert the links as the expected order into the data structure `TrafficFlowModel.__network`
        for origin in origins:
            self.__network.add_origin(origin)
        for destination in destinations:
            self.__network.add_destination(destination)
        
        self.__link_free_time = np.array(link_free_time)
        self.__link_capacity = np.array(link_capacity)
        self.__demand = np.array(demands)
        self.__link_length =np.array(link_length)
        self.__limits = np.array(limits)
        self.__path_flow=np.zeros(76)
        self.__path_length=self.__link_length_to_path_length(self.__link_length)

    def __insert_links_in_order(self, links):
        ''' Insert the links as the expected order into the
            data structure `TrafficFlowModel.__network`
        '''
        first_vertice = [link[0] for link in links]
        for vertex in first_vertice:
            self.__network.add_vertex(vertex)
        for link in links:
            self.__network.add_edge(link)
    
      # Step -1:对比每种车的limit，每对od pair 与最短路；如果 limit大，则保留demand。
    def minlen_to_limit(self):    
        #for i in range(self.length_limits())
            for OD_pair_index in range(self.__network.num_of_OD_pairs()):
                indice_grouped = []
                for path_index in range(self.__network.num_of_paths()):
                    if self.__network.paths_category()[path_index] == OD_pair_index:
                        indice_grouped.append(path_index)  
                        #现在里面有很多条路径哦～
                        sub_path_length = [self.__path_length[ind] for ind in indice_grouped] 
                        ##把该od对上所有的path_time都赋值给sub_path_time哦
                        min_in_group = min(sub_path_length)  
                        #找到最小的距离
                        path_min = sub_path_length.index(min_in_group) 
                        #最小距离对应的索引
                        target_cat_ind = indice_grouped[path_min]
                        #找到全集中的索引
                        #path_minlength[target_cat_ind] = self.__demand[OD_pair_index] #最短路径就被确定了
                        #new_path_length = self.__link_length_to_path_length(path_minlength)  
                        #for j in range(0,self.__num_of_class_of_vehicles()):
                        for j in range(0,2):  
                            if (self.__limits[1, j] -self.__path_length[target_cat_ind] < 0):
                                self.__demand[j,OD_pair_index] = 0


    def solve(self):
        ''' Solve the traffic flow assignment model (user equilibrium)
            by Frank-Wolfe algorithm, all the necessary data must be 
            properly input into the model in advance. 
            如果上述参数都输入了，就开始下一步啦～
            (Implicitly) Return
            ------
            self.__solved = True
        '''
        if self.__detail:  #前文中detail不是被定义为false了吗？？
            print(self.__dash_line()) ##__dash_line就是80个横杆杆
            print("TRAFFIC FLOW ASSIGN MODEL (USER EQUILIBRIUM) \nFRANK-WOLFE ALGORITHM - DETAIL OF ITERATIONS")
            print(self.__dash_line())
            print(self.__dash_line())
            print("Initialization")
            print(self.__dash_line())
        
        # Step 0: based on the x0, generate the x1
        empty_flow = np.zeros(self.__network.num_of_links()) #几条link就几个flow
        link_flow = self.__all_or_nothing_assign(empty_flow) #把空的flow全由全无赋值上去
        
        counter = 0 #计数器
        while True:
            
            if self.__detail:
                print(self.__dash_line())
                print("Iteration %s" % counter)
                print(self.__dash_line())
                print("Current link flow:\n%s" % link_flow)

            # Step 1 & Step 2: Use the link flow matrix -x to generate the time, then generate the auxiliary link flow matrix -y
            auxiliary_link_flow = self.__all_or_nothing_assign(link_flow)
            
            # Step 3: Linear Search 寻找迭代步长。求解一维极小值问题
            opt_theta = self.__golden_section(link_flow, auxiliary_link_flow)
            
            # Step 4: Using optimal theta to update the link flow matrix 更新流量
            new_link_flow = (1 - opt_theta) * link_flow + opt_theta * auxiliary_link_flow

            # Print the detail if necessary
            if self.__detail:
                print("Optimal theta: %.8f" % opt_theta)
                print("Auxiliary link flow:\n%s" % auxiliary_link_flow)

            # Step 5: Check the Convergence, if FALSE, then return to Step 1 收敛性检查。如果满足收敛性准则，则算法终止，
            if self.__is_convergent(link_flow, new_link_flow):
                if self.__detail:
                    print(self.__dash_line())
                self.__solved = True #为了唤醒report
                self.__final_link_flow = new_link_flow
                self.__iterations_times = counter
                break
            else:
                link_flow = new_link_flow
                counter += 1

    def report(self):
        ''' Generate the report of the result in console,
            this function can be invoked only after the
            model is solved.
        '''
        if self.__solved:
            # Print the input of the model
            print(self)
            
            # Print the report
            
            # Do the computation
            link_flow = self.__final_link_flow
            link_time = self.__link_flow_to_link_time(link_flow)
            path_time = self.__link_time_to_path_time(link_time)
            #path_length=self.__link_length_to_path_length(link_length)
            link_vc = link_flow / self.__link_capacity

            print(self.__dash_line())
            print("TRAFFIC FLOW ASSIGN MODEL (USER EQUILIBRIUM) \nFRANK-WOLFE ALGORITHM - REPORT OF SOLUTION")
            print(self.__dash_line())
            print(self.__dash_line())
            print("TIMES OF ITERATION : %d" % self.__iterations_times)
            print(self.__dash_line())
            print(self.__dash_line())
            print("path time: %s" % path_time)
            print("PERFORMANCE OF LINKS")
            print(self.__dash_line())
            for i in range(self.__network.num_of_links()):
                print("%2d : link= %12s, flow= %8.2f, time= %6.2f, v/c= %.3f" % (i, self.__network.edges()[i], link_flow[i], link_time[i], link_vc[i]))
            print(self.__dash_line())
            print("PERFORMANCE OF PATHS (GROUP BY ORIGIN-DESTINATION PAIR)")
            print(self.__dash_line())
            a=np.zeros(12287)
            for i in range(self.__network.num_of_paths()):
                print("%2d : group= %2d, time= %6.2f, path= %s" % (i, self.__network.paths_category()[i], path_time[i], self.__network.paths()[i]))
        #self.__paths_category = self.__generate_paths_by_demands
        else:
            raise ValueError("The report could be generated only after the model is solved!")

    def report_to_excel(self):
        ''' Generate a report of the result to excel file,
            this function can be invoked only after the
            model is solved.
        '''
        if self.__solved:
            
            # Do the computation
            link_flow = self.__final_link_flow
            link_time = self.__link_flow_to_link_time(link_flow)
            path_time = self.__link_time_to_path_time(link_time)
            link_vc = link_flow / self.__link_capacity
            #path_flow = self.__path_flow

            self.__ep.report_to_excel(self.__network.edges(), link_flow, link_time, path_time, link_vc, self.__network.LP_matrix(),self.__path_flow)

            print(self.__dash_line())
            print("TRAFFIC FLOW ASSIGN MODEL (USER EQUILIBRIUM) \nFRANK-WOLFE ALGORITHM - REPORT OF SOLUTION")
            print(self.__dash_line())
            print(self.__dash_line())
            print("TIMES OF ITERATION : %d" % self.__iterations_times)
            print(self.__dash_line())
            print(self.__dash_line())
            print("FOR THE DETAIL PLEASE CHECK OUTPUT.XLS")
            print(self.__dash_line())


    def __all_or_nothing_assign(self, link_flow):
        ''' Perform the all-or-nothing assignment of
            Frank-Wolfe algorithm in the User Equilibrium
            Traffic Assignment Model.
            This assignment aims to assign all the traffic
            flow, within given origin and destination, into
            the least time consuming path

            Input: link flow -> Output: new link flow
            The input is an array.
        '''
        # LINK FLOW -> LINK TIME
        link_time = self.__link_flow_to_link_time(link_flow)
        # LINK TIME -> PATH TIME
        path_time = self.__link_time_to_path_time(link_time)

        # PATH TIME -> PATH FLOW
        # Find the minimal traveling time within group 
        # (splited by origin - destination pairs) and
        # assign all the flow to that path
        self.__path_length = self.__path_length.astype('int16')
        self.__limits = self.__limits.astype('int16')
        self.__path_flow=np.zeros(self.__network.num_of_paths())
        path_flow = np.zeros(self.__network.num_of_paths())
        path_flow_1 = np.zeros(self.__network.num_of_paths())
        path_flow_2 = np.zeros(self.__network.num_of_paths())
        #for j in range(0,self.__num_of_class_of_vehicles()):  ##所有类型的车
        for j in range(0, 2):  ##所有类型的车  
            for OD_pair_index in range(self.__network.num_of_OD_pairs()): #所有的odpair
                indice_grouped = []
                for path_index in range(self.__network.num_of_paths()):   ##所有的路径
                    if self.__network.paths_category()[path_index] == OD_pair_index and self.__path_length[path_index] - self.__limits[1, j] <= 0:
                        indice_grouped.append(path_index)  #现在里面有很多条路径哦～
                sub_path_time = [path_time[ind] for ind in indice_grouped] ##把该od对上所有的path_time都赋值给sub_path_time哦
                min_in_group = min(sub_path_time)  #找到最小的时间
                ind_min = sub_path_time.index(min_in_group) #找到最小时间对应的索引
                target_path_ind = indice_grouped[ind_min]
                path_flow[target_path_ind] += self.__demand[j,OD_pair_index] #那么这条路径上的flow就被确定了
                if j==0:
                    path_flow_1[target_path_ind] += self.__demand[j,OD_pair_index]
                else:
                    path_flow_2[target_path_ind] += self.__demand[j,OD_pair_index]
                #print(self.__path_length[path_index])
                #print(self.__limits[1, j])
                #print(self.__demand[j,OD_pair_index])
                #print(j)
        self.__path_flow=path_flow
                
        if self.__detail:
            print("Link time:\n%s" % link_time)
            print("Path time:\n%s" % path_time)
            print("Path flow:\n%s" % path_flow)
            print("Path flow1:\n%s" % path_flow_1)
            print("Path flow2:\n%s" % path_flow_2)
            
        
        # PATH FLOW -> LINK FLOW
        new_link_flow = self.__path_flow_to_link_flow(path_flow)

        return new_link_flow
        
    def __link_flow_to_link_time(self, link_flow):
        ''' Based on current link flow, use link 
            time performance function to compute the link 
            traveling time.
            The input is an array.
        '''
        n_links = self.__network.num_of_links()
        link_time = np.zeros(n_links)
        for i in range(n_links):
            link_time[i] = self.__link_time_performance(link_flow[i], self.__link_free_time[i], self.__link_capacity[i])
        return link_time

    def __link_time_to_path_time(self, link_time):
        ''' Based on current link traveling time,
            use link-path incidence matrix to compute 
            the path traveling time.
            The input is an array.
        '''
        path_time = link_time.dot(self.__network.LP_matrix())
         ##dot两个数组的点积：linktime点乘linkpath的关系 ##LP_matrix：Link-Path incidence matrix Delta 即路径路段的关系
        return path_time
    
    def __link_length_to_path_length(self, link_length): 
        path_length =  link_length.dot(self.__network.LP_matrix())
        ##dot两个数组的点积：linktime点乘linkpath的关系 ##LP_matrix：Link-Path incidence matrix Delta 即路径路段的关系
        return path_length
    
    def __path_flow_to_link_flow(self, path_flow):
        ''' Based on current path flow, use link-path incidence 
            matrix to compute the traffic flow on each link.
            The input is an array.
        '''
        link_flow = self.__network.LP_matrix().dot(path_flow)
        return link_flow

    def __link_time_performance(self, link_flow, t0, capacity):
        ''' Performance function, which indicates the relationship
            between flows (traffic volume) and travel time on 
            the same link. According to the suggestion from Federal
            Highway Administration (FHWA) of America, we could use
            the following function:
                t = t0 * (1 + alpha * (flow / capacity))^beta
        '''
        value = 1 + link_flow**2
        return value

    def __link_time_performance_integrated(self, link_flow):
        ''' The integrated (with repsect to link flow) form of
            aforementioned performance function.
        '''
        #val1 = t0 * link_flow
        # Some optimization should be implemented for avoiding overflow
        value = link_flow + 1/3*link_flow**3
        return value  

    def __object_function(self, mixed_flow):
        ''' Objective function in the linear search step 
            of the optimization model of user equilibrium 
            traffic assignment problem, the only variable
            is mixed_flow in this case.
        '''
        val = 0
        for i in range(self.__network.num_of_links()):
            val += self.__link_time_performance_integrated(link_flow= mixed_flow[i])
        #print(val)
        return val

    def __golden_section(self, link_flow, auxiliary_link_flow, accuracy= 1e-8):
        ''' The golden-section search is a technique for 
            finding the extremum of a strictly unimodal 
            function by successively narrowing the range
            of values inside which the extremum is known 
            to exist. The accuracy is suggested to be set
            as 1e-8. For more details please refer to:
            https://en.wikipedia.org/wiki/Golden-section_search
        '''
        # Initial params, notice that in our case the
        # optimal theta must be in the i  nterval [0, 1]
        LB = 0
        UB = 1
        goldenPoint = 0.618
        leftX = LB + (1 - goldenPoint) * (UB - LB)
        rightX = LB + goldenPoint * (UB - LB)
        while True:
            val_left = self.__object_function((1 - leftX) * link_flow + leftX * auxiliary_link_flow)
            val_right = self.__object_function((1 - rightX) * link_flow + rightX * auxiliary_link_flow)
            if val_left <= val_right:
                UB = rightX
            else:
                LB = leftX
            if abs(LB - UB) < accuracy:
                opt_theta = (rightX + leftX) / 2.0
                return opt_theta
            else:
                if val_left <= val_right:
                    rightX = leftX
                    leftX = LB + (1 - goldenPoint) * (UB - LB)
                else:
                    leftX = rightX
                    rightX = LB + goldenPoint*(UB - LB)

    def __is_convergent(self, flow1, flow2):
        ''' Regard those two link flows lists as the point
            in Euclidean space R^n, then judge the convergence
            under given accuracy criterion.
            Here the formula
                ERR = || x_{k+1} - x_{k} || / || x_{k} ||
            is recommended.
        '''
        err = np.linalg.norm(flow1 - flow2) / np.linalg.norm(flow1)
        if self.__detail:
            print("ERR: %.8f" % err)
        if err < self._conv_accuracy:
            return True
        else:
            return False
    
    def __num_of_class_of_vehicles(self):
        return len(self.__demand)
    
    def disp_detail(self):
        ''' Display all the numerical details of each variable
            during the iteritions.
        '''
        self.__detail = True

    def set_disp_precision(self, precision):
        ''' Set the precision of display, which influences only
            the digit of numerical component in arrays.
        '''
        np.set_printoptions(precision= precision)

    def __dash_line(self):
        ''' Return a string which consistently 
            contains '-' with fixed length
        '''
        return "-" * 80
    
    def __str__(self):
        string = ""
        string += self.__dash_line()
        string += "\n"
        string += "TRAFFIC FLOW ASSIGN MODEL (USER EQUILIBRIUM) \nFRANK-WOLFE ALGORITHM - PARAMS OF MODEL"
        string += "\n"
        string += self.__dash_line()
        string += "\n"
        string += self.__dash_line()
        string += "\n"
        string += "LINK Information:\n"
        string += self.__dash_line()
        string += "\n"
        for i in range(self.__network.num_of_links()):
            string += "%2d : link= %s, free time= %.2f, capacity= %s \n" % (i, self.__network.edges()[i], self.__link_free_time[i], self.__link_capacity[i])
        string += self.__dash_line()
        string += "\n"
        string += "OD Pairs Information:\n"
        string += self.__dash_line()
        string += "\n"
#        for i in range(self.__network.num_of_OD_pairs()):
#            string += "%2d : OD pair= %s, demand= %d \n" % (i, self.__network.OD_pairs()[i], self.__demand[i])
        string += self.__dash_line()
        string += "\n"
        string += "Path Information:\n"
        string += self.__dash_line()
        string += "\n"
#        for i in range(self.__network.num_of_paths()):
#            string += "%2d : Conjugated OD pair= %s, Path= %s \n" % (i, self.__network.paths_category()[i], self.__network.paths()[i])
        string += self.__dash_line()
        string += "\n"
        string += "Link - Path Incidence Matrix:\n"
        string += self.__dash_line()
        string += "\n"
        string += str(self.__network.LP_matrix())
        return string