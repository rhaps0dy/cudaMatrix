#ifndef _Error_hxx_
#define _Error_hxx_

#include <exception>
#include <cstdlib>

class Error : public std::exception
{
private:
	const char * errStr;
public:
	Error(const char *info = NULL) : errStr(info)
	{
		;
	}
	~Error() throw()
	{
		;
	}

	virtual const char *what() const throw ()
	{
		if (errStr == NULL)
			return "empty";
		return errStr;
	}
};
#endif //_Error_hxx
