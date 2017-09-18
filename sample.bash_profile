################################################################################
#                                                                              #
#            .bash_login file (terminal setup file)                            #
#                                                                              #
#  Note:  This file is only read when bash first starts.                       #
#                                                                              #
#  WARNING:  PLEASE READ THE REST OF THIS FILE BEFORE MAKING ANY CHANGES !!!   #
#                                                                              #
################################################################################

# Get the aliases and functions
if [ -f ~/.bashrc ]; then
	. ~/.bashrc
fi

################################################################################
#                                                                              #
#     If you would like to modify the login initialization sequence, please    #
#     make the changes in a file called .mybashrc located in the top level     #
#     of your home directory.  Your ~/.mybashrc file is read for each bash     #
#     shell started.  For other customizations, please refer to the            #
#     comments in your default ~/.bashrc.                                      #
#                                                                              #
#  WARNING:  Any changes in this file may disappear one day when the system    #
#            is updated!  Make sure you know what you are doing!!!             #
#                                                                              #
################################################################################
export PATH=/opt/anaconda2/bin:/opt/ibm/ILOG/CPLEX_Studio1263/cplex/bin/x86-64_linux:$PATH
export PATH='~/Downloads/Sublime Text 2':$PATH
export PS1="\[\e[0;31m\]\u@\h\[\e[0;36m\]:\[\e[0;34m\]\w \[\e[0;37m\]\n$ \[\e[m\]"
