import numpy as np
import pyfits as pf
import time
from struct import unpack
from Filter import median_filter

def savefits(data, filename, sps=1e6, bits=14, noise=False, clobber=True):
    """
    Save pulse/noise to FITS file
    """
    
    # Prepare data
    data = (np.asarray(data)*2**(bits-1)).round()
    
    # Column Name
    if noise:
    	colname = 'NoiseRec'
    else:
    	colname = 'PulseRec'
        
    # Columns
    col_t = pf.Column(name='TIME', format='1D', unit='s', array=np.zeros(data.shape[0], dtype=int))
    col_data = pf.Column(name=colname, format='%dI' % data.shape[1], unit='V', array=data)
    
    cols = pf.ColDefs([col_t, col_data])
    tbhdu = pf.new_table(cols)
    
    # Name of extension
    exthdr = tbhdu.header
    exthdr.update('EXTNAME', 'Record', 'name of this binary table extension')
    exthdr.update('EXTVER', 1, 'extension version number')
    
    # Add more attributes
    exthdr.update('TSCAL2', 1.0/2**(bits-1), '[V/ch]')
    exthdr.update('TZERO2', 0., '[V]')
    exthdr.update('THSCL2', sps**-1, '[s/bin] horizontal resolution of record')
    exthdr.update('THZER2', 0, '[s] horizontal offset of record')
    exthdr.update('THSAM2', data.shape[1], 'sample number of record')
    exthdr.update('THUNI2', 's', 'physical unit of sampling step of record')
    exthdr.update('TRMIN2', -2**(bits-1)+1, '[channel] minimum number of each sample')
    exthdr.update('TRMAX2', 2**(bits-1)-1, '[channel] maximum number of each sample')
    exthdr.update('TRBIN2', 1, '[channel] default bin number of each sample')
    
    # More attributes
    exthdr.update('TSTART', 0, 'start time of experiment in total second')
    exthdr.update('TSTOP', 0, 'end time of experiment in total second')
    exthdr.update('TEND', 0, 'end time of experiment (obsolete)')
    exthdr.update('DATE', time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()), 'file creation date (UT)')
    
    # We anyway need Primary HDU
    hdu = pf.PrimaryHDU()
    
    # Write to FITS
    thdulist = pf.HDUList([hdu, tbhdu])
    thdulist.writeto(filename, clobber=clobber)

def yopen(filenumber, summary=False, nf=None, tmin=None, tmax=None):
    """
    Read Yokogawa WVF file
    
    Parameters
    ==========
        filenumber: file number to read
        summary:    to summary waves (default: False)
        nf:         sigmas for valid data using median noise filter, None to disable noise filter (default: None)
        tmin:       lower boundary of time for partial extraction, scaler or list (Default: None)
        tmax:       upper boundary of time for partial extraction, scaler or list (Default: None)
    
    Returns
    =======
        if summary is False:
            [ t1, d1, t2, d2, t3, d3, ... ]
        
        if summary is True:
            [ t1, d1, err1, t2, d2, err2, ... ]
        
        where t1 is timing for 1st ch, d1 is data for 1st ch, err1 is error (1sigma) for 1st ch, and so on.
    """
    
    # Read header (HDR)
    h = open(str(filenumber) + ".HDR")
    lines = h.readlines()
    h.close()
    
    # Parse $PublicInfo
    for line in lines:
        token = line.split()
        
        if len(token) > 0:
            # Check endian
            if token[0] == "Endian":
                endian = '>' if token[1] == "Big" else '<'
            
            # Check data format
            if token[0] == "DataFormat":
                format = token[1]
                assert format == "Block"
                
            # Check # of groups
            if token[0] == "GroupNumber":
                groups = int(token[1])
            
            # Check # of total traces
            if token[0] == "TraceTotalNumber":
                ttraces = int(token[1])
            
            # Check data offset
            if token[0] == "DataOffset":
                offset = int(token[1])
    
    # Initialize containers
    traces = [None] * groups        # Number of traces for each group
    blocks = [None] * ttraces       # Number of blocks for each trace
    bsizes = [None] * ttraces       # Block size for each trace
    vres = [None] * ttraces         # VResolution for each trace
    voffset = [None] * ttraces      # VOffset for each trace
    hres = [None] * ttraces         # HResolution for each trace
    hoffset = [None] * ttraces      # HOffset for each trace
    
    # Parse $Group
    for line in lines:
        token = line.split()

        if len(token) > 0:
            # Read current group number
            if token[0][:6] == "$Group":
                cgn = int(token[0][6:]) - 1  # Current group number (minus 1)
            
            # Check # of traces in this group
            if token[0] == "TraceNumber":
                traces[cgn] = int(token[1])
                traceofs = np.sum(traces[:cgn], dtype=int)
                        
            # Check # of Blocks
            if token[0] == "BlockNumber":
                blocks[traceofs:traceofs+traces[cgn]] = [ int(token[1]) ] * traces[cgn]
            
            # Check Block Size
            if token[0] == "BlockSize":
                bsizes[traceofs:traceofs+traces[cgn]] = [ int(s) for s in token[1:] ]
            
            # Check VResolusion
            if token[0] == "VResolution":
                vres[traceofs:traceofs+traces[cgn]] = [ float(res) for res in token[1:] ]
            
            # Check VOffset
            if token[0] == "VOffset":
                voffset[traceofs:traceofs+traces[cgn]] = [ float(ofs) for ofs in token[1:] ]
            
            # Check VDataType
            if token[0] == "VDataType":
                assert token[1] == "IS2"
            
            # Check HResolution
            if token[0] == "HResolution":
                hres[traceofs:traceofs+traces[cgn]] = [ float(res) for res in token[1:] ]
            
            # Check HOffset
            if token[0] == "HOffset":
                hoffset[traceofs:traceofs+traces[cgn]] = [ float(ofs) for ofs in token[1:] ]
        
    # Data Initialization
    time = [ np.array(range(bsizes[t])) * hres[t] + hoffset[t] for t in range(ttraces) ]
    data = [ [None] * blocks[t] for t in range(ttraces) ]
    
    # Open WVF
    f = open(str(filenumber) + ".WVF", 'rb')
    f.seek(offset)
    
    # Read WVF
    if format == "Block":
        # Block format (assuming block size is the same for all the traces in Block format)
        for b in range(blocks[0]):
            for t in range(ttraces):
                data[t][b] = np.array(unpack(endian + 'h'*bsizes[t], f.read(bsizes[t]*2))) * vres[t] + voffset[t]
    else:
        # Trace format
        for t in range(ttraces):
            for b in range(blocks[t]):
                data[t][b] = np.array(unpack(endian + 'h'*bsizes[t], f.read(bsizes[t]*2))) * vres[t] + voffset[t]

    # Array conversion
    for t in range(ttraces):
        data[t] = np.array(data[t])
    
    # Tmin/Tmax filtering
    for t in range(ttraces):
        if type(tmin) == list or type(tmax) == list:
            if not (type(tmin) == list and type(tmax) == list and len(tmin) == len(tmax)):
                raise ValueError("tmin and tmax both have to be list and have to have the same length.")
            mask = np.add.reduce([ (time[t] >= _tmin) & (time[t] < _tmax) for (_tmax, _tmin) in zip(tmax, tmin)], dtype=bool)
        else:
            _tmin = min(time[t]) if tmin is None else tmin
            _tmax = max(time[t]) + 1 if tmax is None else tmax
            mask = (time[t] >= _tmin) & (time[t] < _tmax)
        
        data[t] = data[t][:, mask]
        time[t] = time[t][mask]
        
    f.close()
    
    if summary is False:
        # Return wave data as is
        return [ [ time[t], data[t] ] for t in range(ttraces)  ]
    else:
        if nf is None:
            # Noise filter is off
            return [ [ time[t], np.mean(data[t], axis=0), np.std(data[t], axis=0, ddof=1) ]
                        for t in range(ttraces) ]
        else:
            # Noise filter is on
            return [ [ time[t],
                        np.apply_along_axis(lambda a: np.mean(a[median_filter(a, nf)]), 0,  data[t]),
                        np.apply_along_axis(lambda a: np.std(a[median_filter(a, nf)], ddof=1), 0, data[t]) ]
                            for t in range(ttraces) ]