function greet() {
    console.log("hello");
}

function newFunc(callback) {
    callback()
}

newFunc(greet)

const welcome = () => {
    console.log("hello");
}

newFunc (() => {
    console.log("hello");   
})