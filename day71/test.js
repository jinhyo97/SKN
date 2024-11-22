// let a = 1;
// console.log(a)

// a = 10;
// console.log(a)

// const b = 10;
// console.log(b)

// let score = 11;
// score /= 10;
// let option = parseInt(String(score));

// console.log(option);

// switch (score) {
//     case score >= 90:
//         console.log("A")
//     case 90 > console >=80:
//         console.log("B")
//     case 80 > score >= 70:
//         console.log("C")
//     case 70 > score >= 60:
//         console.log("D")
//     case score < 60:
//         console.log("F")
//     default:
// }

const password = 486486486;
let userInput;

do {
    userInput = prompt("비밀번호를 입력해주세요.");

    if (userInput.length < 10) {
        console.log("비밀번호가 10글자 미만입니다.");
        continue;
    }

    if (userInput == password) {
        console.log("로그인 성공!")
        break;
    }
    else {
        console.log("비밀번호가 일치하지않습니다!");
    }

} while (userInput != password);



class Chobo {
    constructor(str, dex, int, luk, hp, mp) {
        this.str = str
        this.dex = dex
        this.int = int
        this.luk = luk
        this.hp = hp
        this.mp = mp
    }
}