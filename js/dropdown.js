let content = document.getElementsByClassName("dropdown")[0];
function myFunction(obj) {
    console.log(content);
    /*alert(obj.parentNode.parentNode.parentNode.rows.item(2).style.display);
    /*alert(document.getElementById("zh"))
    /*obj.parentNode.parentNode.rowIndex+1*/
    var rowindex = obj.parentNode.parentNode.rowIndex+1;
    if (obj.parentNode.parentNode.parentNode.rows.item(rowindex).style.display == "none") {
        obj.parentNode.parentNode.parentNode.rows.item(rowindex).style.display = "table-row"
        
    } else {
        obj.parentNode.parentNode.parentNode.rows.item(rowindex).style.display = "none" 
    }
    if(obj.classList=="fa fa-angle-right"){
        obj.classList = "fa fa-angle-down";
    }else{
        obj.classList="fa fa-angle-right"
    }
    
    content.classList.add("box-style");
}

function myFunction2(obj) {
    console.log(content);
    var rowindex = obj.parentNode.parentNode.rowIndex+1;
    if (obj.parentNode.parentNode.parentNode.rows.item(rowindex).style.display == "none") {
        obj.parentNode.parentNode.parentNode.rows.item(rowindex).style.display = "table-row"
        
    } else {
        obj.parentNode.parentNode.parentNode.rows.item(rowindex).style.display = "none" 
    }
    if(obj.classList=="fa fa-angle-right"){
        obj.classList = "fa fa-angle-down";
    }else{
        obj.classList="fa fa-angle-right"
    }
    
    content.classList.add("box-style");
}

function myFunction3(obj) {
    console.log(content);
    var rowindex = obj.parentNode.parentNode.rowIndex+1;
    if (obj.parentNode.parentNode.parentNode.rows.item(rowindex).style.display == "none") {
        obj.parentNode.parentNode.parentNode.rows.item(rowindex).style.display = "table-row"
        
    } else {
        obj.parentNode.parentNode.parentNode.rows.item(rowindex).style.display = "none" 
    }
    if(obj.classList=="fa fa-angle-right"){
        obj.classList = "fa fa-angle-down";
    }else{
        obj.classList="fa fa-angle-right"
    }
    
    content.classList.add("box-style");
}

function myFunction4(obj) {
    console.log(content);
    if (document.getElementById("tr4").style.display == "none") {
        document.getElementById("tr4").style.display = "table-row"
        
    } else {
        document.getElementById("tr4").style.display = "none" 
    }
    if(obj.classList=="fa fa-angle-right"){
        obj.classList = "fa fa-angle-down";
    }else{
        obj.classList="fa fa-angle-right"
    }
    
    content.classList.add("box-style");
}