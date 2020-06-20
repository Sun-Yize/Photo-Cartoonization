function varargout = cartoonlize(varargin)
    gui_Singleton = 1;
    gui_State = struct('gui_Name',       mfilename, ...
                       'gui_Singleton',  gui_Singleton, ...
                       'gui_OpeningFcn', @cartoonlize_OpeningFcn, ...
                       'gui_OutputFcn',  @cartoonlize_OutputFcn, ...
                       'gui_LayoutFcn',  [] , ...
                       'gui_Callback',   []);
    if nargin && ischar(varargin{1})
        gui_State.gui_Callback = str2func(varargin{1});
    end

    if nargout
        [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
    else
        gui_mainfcn(gui_State, varargin{:});
    end
% End initialization code - DO NOT EDIT

function TuPianxiansi_OpeningFcn(hObject, eventdata, handles, varargin)
    handles.output = hObject;
    global im; 
    guidata(hObject, handles);

% --- Executes just before cartoonlize is made visible.
function cartoonlize_OpeningFcn(hObject, eventdata, handles, varargin)
    handles.output = hObject;
    guidata(hObject, handles);

% --- Outputs from this function are returned to the command line.
function varargout = cartoonlize_OutputFcn(hObject, eventdata, handles) 
    varargout{1} = handles.output;

% --- Executes on button press in pushbutton2.
function pushbutton2_Callback(hObject, eventdata, handles)
    global img;
    [name,dir,index]=uigetfile({'*.bmp';'*.png'},'Ñ¡ÔñÍ¼Æ¬'); 
    if index==1
    str=[dir name]; 
    img=imread(str); 
    axes(handles.axes1);
    imshow(img);
end
    
function pushbutton1_Callback(hObject, eventdata, handles)
    global img;
    axes(handles.axes1);
    imshow(img);
    img = double(img)/255;
    result = bilateral_filter(double(img),8,[3 0.1]);
    axes(handles.axes2);
    imshow(result);
    result = edge_detect(result);
    axes(handles.axes3);
    imshow(result);
    result = color_adjust(result);
    axes(handles.axes4);
    imshow(result);
    imwrite(result,'result.png')