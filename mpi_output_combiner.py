# Given that MPI processes can output in an uncoordinated fashion (i.e. even when flushing an output line and inserting
# appropriate barriers, you can end up in a situation where the output has yet to flush the output, i.e. due to kernel
# level data structures, even after returning from calls to `std::flush` or adding `std::endl` to a line) that can result
# in the output being out-of-order. Instead, for important information where order is imperative, we have each process write
# to their own file and then concatenate them by timestamp.
#
# The output has different 'sections' which is a single line surrounded in `[` and `]` characters; these must be the same for
# all processes and serve as a kind of barrier and a way to communicating the beginning of a new section. Following this header,
# you have each line of output for that section, ended in a `~[TIMESTAMP]~` which is used to sort the output for the section.
# This script will concatenate and ensure proper ordering based on said timestamps, while outputing headers appropriately.

# Take as command line arguments: --prefix, --output

import sys
import os
import re
import argparse

argparse_obj = argparse.ArgumentParser(description="Combine MPI output files")
argparse_obj.add_argument("--prefix", help="Prefix for output files", required=False, type=str, dest="prefix", default="mpi-proc-")
argparse_obj.add_argument("--output", help="Output file", required=False, type=str, dest="output", default="mpi-output.txt")
args = argparse_obj.parse_args()
prefix = args.prefix
output = args.output

# Read in each file with the prefix
files = [f for f in os.listdir(".") if f.startswith(prefix)]
fds = [open(f) for f in files]
output_fd = open(output, "w")

# Read in, line-by-line, for each file
lines = [fd.readline() for fd in fds]

while True:
    # If a line begins in '[' and ends in ']' then we have a new section
    # However sometimes you have some files that have additional output, so we essentially drain the lines in the file
    # until a section is found; it is an error if a section is found but it is not the same section.
    numWaitForSection = 0
    currSection = None
    minTimestamp = None
    minTimestampIdx = None
    numEmptyLines = 0
    for l,idx in zip(lines,range(len(lines))):
        if l == "": # EOF
            numEmptyLines += 1
            continue
        elif re.match("\[.*\]", l):
            numWaitForSection += 1
            sectionName = l.replace("[","").replace("]","")
            if currSection is not None and currSection != sectionName:
                print(f"Error: Section mismatch; Expected {currSection} but found {sectionName}", file=sys.stderr)
                sys.exit(1)
            currSection = sectionName
        else:
            # Extract timestamp from between two '~' (This is at the very end of the line)
            tmp = re.search("~[0-9]*~\n", l)
            if tmp is None:
                print("Error: No timestamp found in line: " + l)
                exit(-1)
            timestamp = tmp.group(0)[1:-2]
            if minTimestamp is None or timestamp < minTimestamp:
                minTimestamp = timestamp
                minTimestampIdx = idx

    if numWaitForSection == len(fds):
        output_fd.write(currSection)
        lines = [fd.readline() for fd in fds]
    if numEmptyLines == len(fds):
        break
    if numEmptyLines != 0 and numWaitForSection != 0:
        print("Error: Found a section but not all files have a section", file=sys.stderr)
        break
    if minTimestamp is not None:
        # Remove timestamp from line before printing
        outputLine = lines[minTimestampIdx]
        outputLine = re.sub("~[0-9]*~", "", outputLine)
        output_fd.write(outputLine)
        lines[minTimestampIdx] = fds[minTimestampIdx].readline()
