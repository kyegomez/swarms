#!/mnt/data1/swarms/var/swarms/agent_workspace/.venv/bin/python
# -*- coding: utf-8 -*-
import re
import sys
import pdb
import logging
for logger_name in logging.root.manager.loggerDict.keys():
    print(logger_name)
    override_logger = logging.getLogger(logger_name)
for handler in override_logger.handlers:
    print(handler)
    handler.setFormatter(formatter)
    
# from uvicorn.main import main
# if __name__ == '__main__':
#     sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
    
#     try:
#         print("main")
#         pdb.set_trace()
#         ret = main()
#         print(ret)
#     except Exception as e:
#         print(e)
        
        #//    sys.exit(main())
import sys
import uvicorn
from uvicorn.config import LOGGING_CONFIG

def main():
    #root_path = ''
    #if len(sys.argv) >= 2:
    #    root_path = sys.argv[1]
    ##
    # %(name)s : uvicorn, uvicorn.error, ... . Not insightful at all.
    LOGGING_CONFIG["formatters"]["access"]["fmt"] = '%(asctime)s %(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s'
    LOGGING_CONFIG["formatters"]["default"]["fmt"] = "%(asctime)s %(levelprefix)s %(message)s"

    date_fmt = "%Y-%m-%d %H:%M:%S"
    LOGGING_CONFIG["formatters"]["default"]["datefmt"] = date_fmt
    LOGGING_CONFIG["formatters"]["access"]["datefmt"] = date_fmt
    ##
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=7230,
        log_level="trace",
        proxy_headers=True,
        forwarded_allow_ips='*',
        workers=1,
        uds="/mnt/data1/swarms/run/uvicorn/uvicorn-swarms-api.sock")    
    # root_path=root_path                
main()
