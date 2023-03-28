#pragma once

class Listener
{
public:
	virtual void handleEvent(){}
	virtual void handleEvent( int ID ){}
};