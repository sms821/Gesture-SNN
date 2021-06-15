import os
import xlsxwriter

dirname = 'config_800ms_2c_64'
filename = 'del_spikerate_again_99.7.log'
start_line = 45983
end_line = start_line + 10
G = -1
K = 1
if G < 0:
    outfile = 'del_cyc_spikerate.xlsx'
else:
    outfile = 'del_G{}_K{}_spikerate.xlsx'.format(G, K)

def dump_to_excel():
    workbook = xlsxwriter.Workbook( os.path.join(dirname, outfile) )
    worksheet = workbook.add_worksheet()
    # Add a bold format to use to highlight cells.
    bold = workbook.add_format({'bold': True})

    fh = open(os.path.join(dirname, filename))
    row, col = 0, 0
    for i, line in enumerate(fh):
        if start_line <= i <= end_line:
            keys = list(line.split())
            #print(keys)
            for k in keys:
                if k not in [':', ',']:
                    if ':' in k or ',' in k:
                        k = k[:-1]
                    print(row, col, k)
                    if row == 10 and col == 12:
                        item = k
                        worksheet.write(row, col, item, bold)
                    elif row > 0 and col > 0:
                        item = float(k)
                        worksheet.write(row, col, item)
                    else:
                        item = k
                        worksheet.write(row, col, item, bold)
                    col += 1
            row += 1
            col = 0
    workbook.close()

dump_to_excel()
