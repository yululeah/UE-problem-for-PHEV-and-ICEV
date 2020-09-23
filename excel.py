import xlwt
import xlrd
import numpy as np

class ExcelProcessor(object):

    def __init__(self):
        ''' Create an instance of ExcelProcessor class
        '''
        self.__input_filename = "input.xls"
        self.__output_filename = "output.xls"

    def create_template(self):
        ''' Create a template file of input data
            under current work directory
        '''
        workbook = xlwt.Workbook()

        sheet0 = workbook.add_sheet(sheetname= 'BASIC_PARAMS', cell_overwrite_ok= True)
        sheet1 = workbook.add_sheet(sheetname= 'DEMAND1ofveh1', cell_overwrite_ok= True)
        sheet2 = workbook.add_sheet(sheetname= 'DEMAND2ofveh2', cell_overwrite_ok= True)
        sheet3 = workbook.add_sheet(sheetname= 'LIMIT', cell_overwrite_ok= True)

        title_sheet0 = ['From', 'To', 'Free time (km)', 'No. of Lane', 'Free Flow Speed (km/h)', 'Capacity per lane (PCU)']
        for i in range(len(title_sheet0)):
            sheet0.write(0, i, title_sheet0[i])
            
        
        basic_data = [
            
            [1,2,1,1,1,1800],
            [1,3,2,1,1,1800],
            [2,1,3,1,1,1800],
            [2,6,4,1,1,1800],
            [3,1,5,1,1,1800],
            [3,4,6,1,1,1800],
            [3,12,7,1,1,1800],
            [4,3,8,1,1,1800],
            [4,5,9,1,1,1800],
            [4,11,10,1,1,1800],
            [5,4,11,1,1,1800],
            [5,6,12,1,1,1800],
            [5,9,13,1,1,1800],
            [6,2,14,1,1,1800],
            [6,5,15,1,1,1800],
            [6,8,16,1,1,1800],
            [7,8,17,1,1,1800],
            [7,18,18,1,1,1800],
            [8,6,19,1,1,1800],
            [8,7,20,1,1,1800],
            [8,9,21,1,1,1800],
            [8,16,22,1,1,1800],
            [9,5,23,1,1,1800],
            [9,8,24,1,1,1800],
            [9,10,25,1,1,1800],
            [10,9,26,1,1,1800],
            [10,11,27,1,1,1800],
            [10,15,28,1,1,1800],
            [10,16,29,1,1,1800],
            [10,17,30,1,1,1800],
            [11,4,31,1,1,1800],
            [11,10,32,1,1,1800],
            [11,12,33,1,1,1800],
            [11,14,34,1,1,1800],
            [12,3,35,1,1,1800],
            [12,11,36,1,1,1800],
            [12,13,37,1,1,1800],
            [13,12,38,1,1,1800],
            [13,24,39,1,1,1800],
            [14,11,40,1,1,1800],
            [14,15,41,1,1,1800],
            [14,23,42,1,1,1800],
            [15,10,43,1,1,1800],
            [15,14,44,1,1,1800],
            [15,19,45,1,1,1800],
            [15,22,46,1,1,1800],
            [16,8,47,1,1,1800],
            [16,10,48,1,1,1800],
            [16,17,49,1,1,1800],
            [16,18,50,1,1,1800],
            [17,10,51,1,1,1800],
            [17,16,52,1,1,1800],
            [17,19,53,1,1,1800],
            [18,7,54,1,1,1800],
            [18,16,55,1,1,1800],
            [18,20,56,1,1,1800],
            [19,15,57,1,1,1800],
            [19,17,58,1,1,1800],
            [19,20,59,1,1,1800],
            [20,18,60,1,1,1800],
            [20,19,61,1,1,1800],
            [20,21,62,1,1,1800],
            [20,22,63,1,1,1800],
            [21,20,64,1,1,1800],
            [21,22,65,1,1,1800],
            [21,24,66,1,1,1800],
            [22,15,67,1,1,1800],
            [22,20,68,1,1,1800],
            [22,21,69,1,1,1800],
            [22,23,70,1,1,1800],
            [23,14,71,1,1,1800],
            [23,22,72,1,1,1800],
            [23,24,73,1,1,1800],
            [24,13,74,1,1,1800],
            [24,21,75,1,1,1800],
            [24,23,76,1,1,1800]

        ]
        for row in range(1, len(basic_data)+1):
            for col in range(len(basic_data[0])):
                sheet0.write(row, col, basic_data[row-1][col])

        sheet1.write(0, 0, 'Demand')
        sheet1.write(0, 1, 10)
        sheet1.write(0, 2, 20)
        sheet1.write(1, 0, 1)
        sheet1.write(2, 0, 2)
        demands = [
            [200, 0],
            [0, 0]
        ]
        for i in range(1, len(demands)+1):
            for j in range(1, len(demands)+1):
                sheet1.write(i, j, demands[i-1][j-1])

        sheet2.write(0, 0, 'Demand')
        sheet2.write(0, 1, 10)
        sheet2.write(0, 2, 20)
        sheet2.write(1, 0, 1)
        sheet2.write(2, 0, 2)
        demands = [
            [0, 0],
            [0, 0]
        ]
        for i in range(1, len(demands)+1):
            for j in range(1, len(demands)+1):
                sheet2.write(i, j, demands[i-1][j-1])

        
        sheet3.write(0, 0, 'class_of_vehicle')
        sheet3.write(1, 0, 'limit')
        #sheet3.write(0, 1, 1) #第一种车
        #sheet3.write(1, 1, 27) #limit是27

        limits = [
            [1,2],  #车型
            [999,999] #限制
        ]
        for i in range(1, len(limits)+1):
            for j in range(1, len(limits)+1):
                sheet3.write(i-1, j, limits[i-1][j-1])
                
        workbook.save(self.__input_filename)

    def read_links(self):
        ''' Read the topology of the graph
            from the file `input.xls`
        '''
        workbook = xlrd.open_workbook(self.__input_filename)
        data = workbook.sheet_by_index(0)
        rows = data.nrows
        links = [ [self.__trans(data.cell(i, 0).value), self.__trans(data.cell(i, 1).value)] for i in range(1, rows) ]
        return links

    def read_basic_params(self):
        ''' Read the values of variables link_free_time
            and link_capacity from the file `input.xls`
        '''
        workbook = xlrd.open_workbook(self.__input_filename)
        data = workbook.sheet_by_index(0)
        rows = data.nrows

        distance = [float(data.cell(row, 2).value) for row in range(1, rows)]
        link_capacity = [float(data.cell(row, 3).value * float(data.cell(row, 5).value)) for row in range(1, rows)]
        #free_speed = [float(data.cell(row, 4).value) for row in range(1, rows)]
        link_free_time = [distance[i] for i in range(len(link_capacity))]

        return distance, link_free_time,link_capacity#link_free_time,
    
    def read_limits(self):
        ''' Read the values of variables link_free_time
            and link_capacity from the file `input.xls`
        '''
        workbook = xlrd.open_workbook(self.__input_filename)
        data = workbook.sheet_by_index(3)
        cols = data.ncols

        vehicle_class = [self.__trans(data.cell(0, j).value) for j in range(1, cols)]
        vehicle_limit = [self.__trans(data.cell(1, j).value) for j in range(1, cols)]
        
        return vehicle_class, vehicle_limit

    def read_demands(self):
        ''' Read the values of variable origins, 
            destinations and demands from the
            file `input.xls`
        ''' 
        '''
            记得修改尺寸大小
        '''
        demands = np.zeros((2,4), dtype=np.int16) 
        workbook = xlrd.open_workbook(self.__input_filename)
        demand1 = workbook.sheet_by_index(1)
        rows = demand1.nrows
        cols = demand1.ncols

        origins = [self.__trans(demand1.cell(i, 0).value) for i in range(1, rows)]
        destinations = [self.__trans(demand1.cell(0, j).value) for j in range(1, cols)]
        demands1 = [demand1.cell(i, j).value for i in range(1, rows) for j in range(1, cols)]
        demands[0,:] = demands1
        
        demand2 = workbook.sheet_by_index(2)
        rows = demand2.nrows
        cols = demand2.ncols

        demands2 = [demand2.cell(i, j).value for i in range(1, rows) for j in range(1, cols)]
        demands[1,:]=demands2
        
        return origins, destinations, demands

    def report_to_excel(self, links, link_flow, link_time, path_time, link_vc, LP_matrix,path_flow):
        ''' Interface between Python and Excel, 
            used for generating the solution report
        '''
#        path_flow = np.transpose(LP_matrix).dot(link_flow)
#        for i in range(len(LP_matrix)):
#            if min(link_flow(LP_matrix[i]))==0:
#                Path_flow[i]=0
#            else:
#                Path_flow=1
        #path_time = self.__link_time_to_path_time(link_time)
        #path_flow =self.__path_flow
        
        workbook = xlwt.Workbook()
        flow_sheet = workbook.add_sheet(sheetname= 'FLOW', cell_overwrite_ok= True)
        graph_sheet = workbook.add_sheet(sheetname= 'GRAPH', cell_overwrite_ok= True)

        title1 = ['No.', 'Origin', 'Destination', 'Link Flow', 'Link Time', 'V/C']
        width1 = len(title1) + 1

        for i in range(len(title1)):
            flow_sheet.write(0, i, title1[i])

        for row in range(1, len(links)+1):
            flow_sheet.write(row, 0, row)
            flow_sheet.write(row, 1, links[row-1][0])
            flow_sheet.write(row, 2, links[row-1][1])
            flow_sheet.write(row, 3, round(link_flow[row-1], 3))
            flow_sheet.write(row, 4, round(link_time[row-1], 3))
            flow_sheet.write(row, 5, round(link_vc[row-1], 3))

        title2 = ['No.', 'Path Time']
        width2 = width1 + len(title2) + 1

        for i in range(len(title2)):
            flow_sheet.write(0, i + width1, title2[i])

        for row in range(1, path_time.shape[0] + 1):
            flow_sheet.write(row, 0 + width1, row)
            flow_sheet.write(row, 1 + width1, round(path_time[row-1], 3))
        
        title3 = ['No.', 'Path flow']
        width3 = width2 + len(title3) + 1

        for i in range(len(title3)):
            flow_sheet.write(0, i + width2, title3[i])

        for row in range(1, path_flow.shape[0] + 1):
            flow_sheet.write(row, 0 + width2, row)
            flow_sheet.write(row, 1 + width2, round(path_flow[row-1], 3))
            
#        graph_sheet.write(0, 0, 'LP Matrix')
#
#        for i in range(LP_matrix.shape[0]):
#            graph_sheet.write(i+1, 0, i+1)
#
#        for j in range(LP_matrix.shape[1]):
#            graph_sheet.write(0, j+1, j+1)
#
#        for i in range(LP_matrix.shape[0]):
#            for j in range(LP_matrix.shape[1]):
#                graph_sheet.write(i+1, j+1, int(LP_matrix[i,j]))

        workbook.save(self.__output_filename)
    
    def __trans(self, num):
        """ Try to transform a float into int, and then
            into string type. If input is not float, it
            will be returned back without any change
        """
        if isinstance(num, float):
            return str(int(num))
        else:
            return num
        