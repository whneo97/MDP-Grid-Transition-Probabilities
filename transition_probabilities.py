import os, webbrowser

class Probabilities_Grid():
    class __Node():
        def __init__(self, cell):
            self.cell = cell
            self.children = {}
            self.parent = None

        def addChild(self, child, probability):
            self.children[child] = probability
            child.parent = self, probability
    
    def __init__(self, rows, cols, terminal_states, blocked_states, init_cell, mv_ls, mv_map):
        self.rows, self.cols = rows, cols
        self.terminal_states = terminal_states
        self.blocked_states = blocked_states
        self.init_cell = init_cell
        self.mv_ls = mv_ls
        self.mv_map = mv_map
        self.__grid = [[1 for j in range(self.cols)] for i in range(self.rows)]
        self.__probabilities = [[0 for j in range(self.cols)] for i in range(self.rows)]
        for x, y in self.blocked_states:
            self.__grid[y-1][x-1] = 0
        self.__node = self.__Node(self.init_cell)
        self.__propagate(self.__node, self.mv_ls)
        self.__fill_probabilities(self.__node)
        self.output_to_html(self.__get_puzzle_grid_string(), 'puzzle_grid')
        self.output_to_html(self.__get_probabilities_grid_string(), 'probabilities_grid')
        self.output_to_html(self.__get_tree_string(self.__node), 'probabilities_tree')
        
    def __move(self, curr, mv):
        x, y = curr
        def get_next_x_y():
            if mv == 'UP': return x, y + 1
            if mv == 'DOWN': return x, y - 1
            if mv == 'RIGHT': return x + 1, y
            if mv == 'LEFT': return x - 1, y
            assert False
        assert x > 0 and y > 0, print(x, y)
        next_x, next_y = get_next_x_y()  
        if next_x > cols or next_x < 1: return curr
        if next_y > rows or next_y < 1: return curr
        if self.__grid[next_y - 1][next_x - 1] == 0: return curr
        return next_x, next_y

    def __propagate(self, node, mv_ls):
        mv_ls = mv_ls.copy()
        if len(mv_ls) != 0 and node.cell not in terminal_states:
            next_mv = mv_ls.pop(0)
            for prob, mv in mv_map[next_mv]:
                child = self.__Node(self.__move(node.cell, mv))
                node.addChild(child, prob)
                self.__propagate(child, mv_ls)

    def __fill_probabilities(self, node):
        def get_probability(leaf):
            if leaf.parent is None: return 1       
            parent, prob = leaf.parent
            return prob * get_probability(parent)
        if len(node.children) != 0:
            for child, prob in node.children.items():
                self.__fill_probabilities(child)
        else:
            x, y  = node.cell
            self.__probabilities[y-1][x-1] += get_probability(node)

    def __get_puzzle_grid_string(self):
        string = ''
        grid = [[' '*5 for j in range(self.cols)] for i in range(self.rows)]
        for i, j in self.terminal_states:
            grid[j-1][i-1] = 'TERM '
        for i, j in self.blocked_states:
            grid[j-1][i-1] = '  X  '
        i, j = self.init_cell
        grid[j-1][i-1] = 'START'
        for i in range(self.rows - 1, -1, -1):
            row_contents = '| '
            row_contents += ' | '.join([grid[i][j] for j in range(self.cols)])
            row_contents += ' |'
            string += '\n'.join([row_contents, '-' * len(row_contents), ''])
        if len(row_contents.split('\n')) != 0:
            string = '-' * len(row_contents.split('\n')[0]) + '\n' + string
        return string.strip()

    def __get_probabilities_grid_string(self):
        string = ''
        grid = self.__probabilities
        for i in range(self.rows - 1, -1, -1):
            row_contents = '| '
            row_contents += ' | '.join(['{item:.5f}'.format(item=grid[i][j]) for j in range(self.cols)])
            row_contents += ' |'
            string += '\n'.join([row_contents, '-' * len(row_contents), ''])
        if len(row_contents.split('\n')) != 0:
            string = '-' * len(row_contents.split('\n')[0]) + '\n' + string
        return string.strip()

    def __get_tree_string(self, node):
        def get_block_with_standardised_width(block):
            block_width = max([len(line) for line in block.split('\n')])
            return '\n'.join([line.ljust(block_width) for line in block.split('\n') if line.strip() != ''])

        def get_width(string):
            return max(len(line) for line in string.split('\n'))

        def get_centralised_string(string, length, fill=' '):
            if len(string) == length: return string
            midpoint = length // 2
            midpoint_string = len(string) // 2
            offset = max(midpoint - midpoint_string, 0)
            centralised_string = fill * offset + string
            return centralised_string

        def get_vertically_connected_block(top, bottom, sep='|', sep_text=''):
            width = max(get_width(top), get_width(bottom))
            connected = '\n'.join([get_centralised_string(top, width), 
                                   get_centralised_string(sep, width) + ' ' + sep_text, 
                                   get_centralised_string(bottom, width)])
            return get_block_with_standardised_width(connected)

        def get_horizontally_concatenated_blocks(blocks, sep=' '):
            if len(blocks) ==  0: return ''
            if len(blocks) == 1: return blocks[0]
            def get_concatenated_pair(block1, block2):
                if len(block1) == 0: return block2
                if len(block2) == 0: return block1
                block1 = get_block_with_standardised_width(block1)
                block2 = get_block_with_standardised_width(block2)
                block1_lines, block2_lines = block1.split('\n'), block2.split('\n')
                width_block1, width_block2 = len(block1_lines[0]), len(block2_lines[0])
                num_lines = max(len(block1_lines), len(block2_lines))
                block1_extended = [(width_block1 * ' ') if i >= len(block1_lines) else block1_lines[i] for i in range(num_lines)]
                block2_extended = [(width_block2 * ' ') if i >= len(block2_lines) else block2_lines[i] for i in range(num_lines)]
                return '\n'.join([line1 + sep + line2 for line1, line2 in zip(block1_extended, block2_extended)])
            curr_block = blocks[0]
            for next_block in blocks[1:]:
                curr_block = get_concatenated_pair(curr_block, next_block)
            return curr_block

        def get_horizontal_bar(block, sep='|'):
            block = get_block_with_standardised_width(block)
            lines = block.split('\n')
            if len(lines) == 0: return ''
            if lines[0].count(sep) < 2: return ''
            width = len(lines[0])
            left, right = lines[0].find(sep), lines[0].rfind(sep)
            return ''.join(['-' if left <= i <= right else ' ' for i in range(width)])

        def get_children_with_stems(children):
            return [get_vertically_connected_block('', self.__get_tree_string(child), sep_text=str(round(prob, 2))) 
                    for child, prob in children.items()]

        if len(node.children) == 0: return str(node.cell)
        child_strings = get_children_with_stems(node.children)
        if len(child_strings) == 1: return get_vertically_connected_block(str(node.cell), child_strings[0].cell, sep='')
        concatenated_children = get_horizontally_concatenated_blocks(child_strings)
        concatenated_children_with_bar = get_horizontal_bar(concatenated_children) + '\n' + concatenated_children
        return get_vertically_connected_block(str(node.cell), concatenated_children_with_bar)
    
    def output_to_html(self, string, name):
        output_folder = 'probabilties_outputs'
        filename = f'{output_folder}/{name}.html'
        if not os.path.exists(output_folder): os.makedirs(output_folder)
        if os.path.isfile(filename):
            name = name + '_copy'
            filename = f'{output_folder}/{name}.html'            
        with open(f'{output_folder}/{name}.html','w') as f:                
            string = string.replace('\n', '<br>')
            html_output = f"""
            <!DOCTYPE html>
            <html>
            <body>

            <div style="white-space: pre; font-family: 'Courier New', monospace">{string}</div>

            </body>
            </html>
            """
            f.write(html_output)
            webbrowser.open('file://' + os.path.realpath(filename), new=2)

rows, cols = 3, 4
terminal_states = [(4, 2), (4, 3)]
blocked_states = [(2, 2)]
init_cell = (1, 1)
mv_ls = ['RIGHT', 'RIGHT', 'RIGHT', 'UP', 'UP']
mv_map = {'UP': [(0.8, 'UP'), (0.1, 'LEFT'), (0.1, 'RIGHT')], 
          'DOWN': [(0.8, 'DOWN'), (0.1, 'LEFT'), (0.1, 'RIGHT')], 
          'LEFT': [(0.8, 'LEFT'), (0.1, 'UP'), (0.1, 'DOWN')],
          'RIGHT': [(0.8, 'RIGHT'), (0.1, 'UP'), (0.1, 'DOWN')]}

Probabilities_Grid(rows, cols, terminal_states, blocked_states, init_cell, mv_ls, mv_map)

# rows, cols = 3, 3
# terminal_states = [(3, 1)]
# blocked_states = []
# init_cell = (2, 2)
# mv_ls = ['DOWN', 'RIGHT']
# mv_map = {'UP': [(0.25, 'UP'), (0.25, 'DOWN'), (0.25, 'LEFT'), (0.25, 'RIGHT')], 
#           'DOWN': [(0.25, 'UP'), (0.25, 'DOWN'), (0.25, 'LEFT'), (0.25, 'RIGHT')], 
#           'LEFT': [(0.25, 'UP'), (0.25, 'DOWN'), (0.25, 'LEFT'), (0.25, 'RIGHT')],
#           'RIGHT': [(0.25, 'UP'), (0.25, 'DOWN'), (0.25, 'LEFT'), (0.25, 'RIGHT')]}

# Probabilities_Grid(rows, cols, terminal_states, blocked_states, init_cell, mv_ls, mv_map)

