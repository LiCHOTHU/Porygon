import os
# import subprocess



def main():
    job_ids = [
        4134113,
        4134114,
        4134115
        
        
    ]

    for job_id in job_ids:
        path = f'slurm_out/Report-{job_id}.out'
        # print(path)
        with open(path, 'r') as f:
            lines = []
            for i, line in enumerate(f):
                lines.append(line)
            
                if i > 100:
                    break
            cmd = lines[lines.index('Running the following command:\n') + 1]
            # print(cmd)
            os.system(f'sbatch slurm/run_l40s.sbatch {cmd}')
            # args = cmd.split(' ')
            # arg_dict = {}
            # for arg in args:
            #     if len(arg.split('=')) == 1:
            #         continue
            #     key, value = arg.split('=')
            #     arg_dict[key] = value
            # # print(arg_dict)
            # if kill_fn(arg_dict):
            #     print(f' {job_id}')
                # print(arg_dict['variant_name'])
            # else:
            #     print('SAFE:', arg_dict['variant_name'])
                
        # for i, x in enumerate(f):
        #     print(x)
        #     if i > 10:
        #         break
        
if __name__ == '__main__':
    main()
    
